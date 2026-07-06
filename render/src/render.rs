//! The wgpu splat renderer: [`Renderer`] holds the reusable GPU context + pipeline,
//! [`Renderer::render_frame`] rasterizes one frame of Contract 3 frame-data into a
//! linear HDR [`HdrImage`].
//!
//! Each particle is drawn as an instanced quad whose fragment applies a Gaussian
//! falloff, additively blended (`src·1 + dst·1`) into an `Rgba32Float` target — the
//! order-independent accumulation DESIGN calls for. Instances carry **world-space**
//! position/radius; the camera (basis + projection parameters) is a uniform and
//! projection happens in the vertex shader (M6g — the 10⁸-particle path: no
//! per-frame CPU projection loop). Orthographic reproduces the retired CPU
//! projection bit-for-bit in formula (pinned by the golden gate in
//! `tests/vertex_path.rs`); perspective keeps peak surface intensity fixed and
//! shrinks screen size ∝ 1/depth, so apparent flux follows the physical 1/d² law
//! with no tuned attenuation factor. The GPU context is built once and reused.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use galaxy_renderprep::FrameData;

use crate::camera::{Camera, Projection};
use crate::RenderError;

/// HDR accumulation format: 32-bit float so galaxy cores don't saturate/band (16F
/// is explicitly rejected in DESIGN). Additive blend into it needs FLOAT32_BLENDABLE.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Per-frame render settings. Camera lives separately (it changes per view, not per
/// frame); this is the raster target + splat shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Gaussian falloff constant `k`: a splat's intensity is `exp(-k · r²)` for `r`
    /// the normalized distance (0 at center, 1 at the quad edge). Larger = tighter.
    pub falloff: f32,
    /// Perspective only: minimum on-screen splat half-extent in *pixels*. A splat
    /// whose projected size falls below this is drawn at this size with its
    /// emission dimmed by (true/clamped)² — the point-source regime: integrated
    /// flux keeps the physical 1/d² law while distant stars stop shimmering as
    /// sub-pixel quads. Ignored by orthographic cameras (bit-compat).
    pub min_splat_px: f32,
    /// Perspective only: maximum splat half-extent in NDC units, guarding fill
    /// rate on close fly-bys. Clamping *down* does NOT boost emission — apparent
    /// flux saturates instead of diverging as depth → near. Ignored by
    /// orthographic cameras.
    pub max_splat_ndc: f32,
    /// **Both projections**: maximum on-screen splat half-extent in *pixels* —
    /// the screen-space PSF of a point source, so stars stay point-like at any
    /// zoom (docs/plans/pinprick-starfield.md). Clamping down boosts emission
    /// by (true/clamped)², the exact mirror of the `min_splat_px` dimming:
    /// integrated flux is invariant, the cap only reshapes the PSF. `INFINITY`
    /// (the default) = off, bit-identical to the uncapped render. A finite cap
    /// must be > 0, and under perspective ≥ `min_splat_px` (a crossed clamp
    /// window is a config error). The `max_splat_ndc` fill-rate guard stays
    /// outermost and saturating.
    pub max_splat_px: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 1920,
            height: 1080,
            falloff: 6.0,
            min_splat_px: 1.5,
            max_splat_ndc: 1.0,
            max_splat_px: f32::INFINITY,
        }
    }
}

impl RenderConfig {
    /// The image aspect ratio (width / height), for camera framing.
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// A linear HDR image: `width × height` RGBA pixels, 32-bit float, row-major from
/// the top-left. Not tonemapped — this is what `grade` consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `width * height` RGBA pixels, row-major.
    pub pixels: Vec<[f32; 4]>,
}

impl HdrImage {
    /// The pixel at `(x, y)` (top-left origin).
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 4] {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Sum of each RGB channel over all pixels — the total accumulated flux, used by
    /// conservation/linearity invariants.
    pub fn total_flux(&self) -> [f64; 3] {
        let mut sum = [0.0f64; 3];
        for p in &self.pixels {
            sum[0] += p[0] as f64;
            sum[1] += p[1] as f64;
            sum[2] += p[2] as f64;
        }
        sum
    }
}

/// One splat as uploaded to the GPU: **world-space** position and radius plus
/// premultiplied emissive color (`color · brightness`). Projection is the vertex
/// shader's job — the instance buffer is camera-independent.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplat {
    pos: [f32; 3],
    radius: f32,
    emissive: [f32; 3],
    _pad: f32,
}

/// Per-frame uniform: camera basis + projection parameters + splat-clamp policy.
/// All vec4-aligned; the layout mirrors the WGSL `Uniforms` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    /// Screen-right axis (xyz; w unused).
    right: [f32; 4],
    /// Screen-up axis (xyz; w unused).
    up: [f32; 4],
    /// View direction into the screen (xyz; w unused).
    forward: [f32; 4],
    /// World-space view target (xyz; w unused).
    target: [f32; 4],
    /// x, y: half_extent at the target plane; z: eye distance; w: near depth
    /// (z, w meaningful for perspective only).
    view: [f32; 4],
    /// x: projection mode (0 = ortho, 1 = perspective); y: Gaussian falloff;
    /// z: min splat half-extent in pixels; w: max splat half-extent in NDC.
    params: [f32; 4],
    /// x, y: viewport half-width / half-height in pixels (NDC→px scale).
    viewport: [f32; 4],
}

/// The camera/config uniform block and its binding — shared verbatim by the
/// star, gas, and prepass shaders (one Rust-side buffer serves all three).
const WGSL_UNIFORMS: &str = r#"
struct Uniforms {
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    // `target` is a reserved WGSL keyword; same slot as Uniforms::target.
    view_target: vec4<f32>,
    view: vec4<f32>,
    params: vec4<f32>,
    viewport: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
"#;

const SHADER: &str = r#"
// Per-star transmittance from the M7e compute prepass (1.0 everywhere when
// gas is off — ×1.0 is bit-exact, pinning the M6g golden).
@group(1) @binding(0) var<storage, read> star_t: array<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) emissive: vec3<f32>,
};

// Degenerate clip position: z > w, so the whole primitive is discarded before
// rasterization. Used to cull at/behind-near splats without touching the 1/z pole.
fn culled() -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.local = vec2<f32>(0.0, 0.0);
    out.emissive = vec3<f32>(0.0, 0.0, 0.0);
    return out;
}

@vertex
fn vs(@builtin(instance_index) instance: u32,
      @location(0) corner: vec2<f32>,
      @location(1) world: vec3<f32>,
      @location(2) radius: f32,
      @location(3) emissive: vec3<f32>) -> VsOut {
    let d = world - u.view_target.xyz;
    let lateral = vec2<f32>(dot(d, u.right.xyz), dot(d, u.up.xyz));
    let he = u.view.xy;

    var ndc: vec2<f32>;
    var half: vec2<f32>;
    var dim = 1.0;
    if (u.params.x < 0.5) {
        // Orthographic: the exact arithmetic of the retired CPU projection
        // (golden-gated), position-independent splat size, no clamps.
        ndc = lateral / he;
        half = vec2<f32>(radius, radius) / he;
    } else {
        // Perspective: similar triangles about the pinhole at depth `distance`
        // behind the target. At/behind the near plane the whole quad is culled
        // (splats have no depth extent) and the 1/z pole is never evaluated.
        let z = dot(d, u.forward.xyz) + u.view.z;
        if (z <= u.view.w) {
            return culled();
        }
        let s = u.view.z / z;
        ndc = lateral * s / he;
        half = vec2<f32>(radius, radius) * s / he;

        // Pixel-space size clamp (aspect-correct cameras keep splats isotropic
        // on screen; the y axis is the scalar). Clamping UP from sub-pixel dims
        // emission by (true/clamped)^2 — the point-source regime, flux keeps
        // the physical 1/d^2 law. Clamping DOWN (fill-rate guard) saturates:
        // no brightness boost.
        let py = half.y * u.viewport.y;
        if (py <= 0.0) {
            return culled();
        }
        let py_clamped = clamp(py, u.params.z, u.params.w * u.viewport.y);
        let scale = py_clamped / py;
        half = half * scale;
        dim = min(1.0, 1.0 / (scale * scale));
    }

    var out: VsOut;
    out.pos = vec4<f32>(ndc + corner * half, 0.0, 1.0);
    out.local = corner;
    // Gas attenuation dims the emission only (alpha stays the splat weight).
    out.emissive = emissive * dim * star_t[instance];
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let g = exp(-u.params.y * dot(in.local, in.local));
    return vec4<f32>(in.emissive * g, g);
}
"#;

/// Gas-side WGSL shared by the fullscreen gas pass and the transmittance
/// prepass: the gas uniform block, the two endpoint density textures, the AABB
/// clip, and manual-trilinear sampling. Every function mirrors its CPU
/// reference in [`crate::volume`] operation-for-operation — sampling is
/// deliberately `textureLoad`-based (8 fetches, exact f32 arithmetic) rather
/// than a FLOAT32_FILTERABLE sampler, whose fixed-point subtexel weights would
/// make the GPU ≡ CPU gates hardware-dependent (the filtered fast path is a
/// named deferral).
const WGSL_GAS_COMMON: &str = r#"
struct GasUniforms {
    // xyz: emission color tint; w: emissivity j.
    ce: vec4<f32>,
    // x: opacity kappa; y: endpoint mix u; z: nominal step; w: unused.
    kms: vec4<f32>,
    b0min: vec4<f32>,
    b0max: vec4<f32>,
    b1min: vec4<f32>,
    b1max: vec4<f32>,
    // Union AABB of both grids: the march domain.
    mmin: vec4<f32>,
    mmax: vec4<f32>,
    // Single-scatter starlight: x = strength sigma_s, y = HG anisotropy g,
    // z = light count (0 disables — bit-compat off path), w unused.
    scat: vec4<f32>,
};
@group(1) @binding(0) var<uniform> g: GasUniforms;
@group(1) @binding(1) var rho0: texture_3d<f32>;
@group(1) @binding(2) var rho1: texture_3d<f32>;

// Slab clip of origin + t*dir against [bmin, bmax]; returns (t0, t1), with
// t0 >= t1 encoding a miss. Mirrors volume::clip_aabb (same +-1e30 sentinels,
// same near-zero-axis inside test).
fn clip_aabb(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    var t0 = -1e30;
    var t1 = 1e30;
    for (var a = 0; a < 3; a++) {
        if (abs(d[a]) < 1e-12) {
            if (o[a] < bmin[a] || o[a] > bmax[a]) {
                return vec2<f32>(1.0, -1.0);
            }
        } else {
            let ta = (bmin[a] - o[a]) / d[a];
            let tb = (bmax[a] - o[a]) / d[a];
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
    }
    return vec2<f32>(t0, t1);
}

// Manual trilinear sample of one grid: exact cell values at cell centers,
// clamp-to-edge in the outer half-cell ring, zero outside the bounds.
// Mirrors GasGrid::sample (the documented shader oracle).
fn sample_one(t: texture_3d<f32>, bmin: vec3<f32>, bmax: vec3<f32>, p: vec3<f32>) -> f32 {
    if (any(p < bmin) || any(p > bmax)) {
        return 0.0;
    }
    let dims = vec3<f32>(textureDimensions(t));
    let cell = (bmax - bmin) / dims;
    let c = (p - bmin) / cell - vec3<f32>(0.5);
    let maxi = dims - vec3<f32>(1.0);
    let cc = clamp(c, vec3<f32>(0.0), maxi);
    let i0 = max(min(floor(cc), maxi - vec3<f32>(1.0)), vec3<f32>(0.0));
    let i1 = min(i0 + vec3<f32>(1.0), maxi);
    let fr = cc - i0;
    let a = vec3<u32>(i0);
    let b = vec3<u32>(i1);
    let c000 = textureLoad(t, vec3<u32>(a.x, a.y, a.z), 0).x;
    let c100 = textureLoad(t, vec3<u32>(b.x, a.y, a.z), 0).x;
    let c010 = textureLoad(t, vec3<u32>(a.x, b.y, a.z), 0).x;
    let c110 = textureLoad(t, vec3<u32>(b.x, b.y, a.z), 0).x;
    let c001 = textureLoad(t, vec3<u32>(a.x, a.y, b.z), 0).x;
    let c101 = textureLoad(t, vec3<u32>(b.x, a.y, b.z), 0).x;
    let c011 = textureLoad(t, vec3<u32>(a.x, b.y, b.z), 0).x;
    let c111 = textureLoad(t, vec3<u32>(b.x, b.y, b.z), 0).x;
    // Two-product lerps, bit-exact at fr = 0 and fr = 1 (GasGrid::sample's rule).
    let c00 = (1.0 - fr.x) * c000 + fr.x * c100;
    let c10 = (1.0 - fr.x) * c010 + fr.x * c110;
    let c01 = (1.0 - fr.x) * c001 + fr.x * c101;
    let c11 = (1.0 - fr.x) * c011 + fr.x * c111;
    let c0 = (1.0 - fr.y) * c00 + fr.y * c10;
    let c1 = (1.0 - fr.y) * c01 + fr.y * c11;
    return (1.0 - fr.z) * c0 + fr.z * c1;
}

// The endpoint-mixed density: mirrors renderprep's sample_mix two-product lerp.
fn density_at(p: vec3<f32>) -> f32 {
    return (1.0 - g.kms.y) * sample_one(rho0, g.b0min.xyz, g.b0max.xyz, p)
        + g.kms.y * sample_one(rho1, g.b1min.xyz, g.b1max.xyz, p);
}
"#;

/// The fullscreen gas pass: per-pixel camera ray (the splat path's NDC
/// convention), union-AABB clip, front-to-back midpoint march with early exit
/// (volume.rs march rule verbatim), additively blended `(radiance, 1 − T)`.
/// `{exit}` / `{max_steps}` are injected from the `volume` constants.
const WGSL_GAS_PASS: &str = r#"
// Point-light proxies for the single-scatter term (volume::Light, clustered
// CPU-side by cluster_lights). Fragment-only: the prepass shares the bind
// group layout but never reads them.
struct PointLight {
    pos: vec3<f32>,
    radius: f32,
    rgb: vec3<f32>,
    pad: f32,
};
@group(1) @binding(3) var<storage, read> lights: array<PointLight>;

// Henyey-Greenstein phase, mirroring volume::hg_phase (which evaluates in f64;
// the GPU == CPU gates allow the f32 difference). 12.566... = 4*pi.
fn hg_phase(mu: f32, ga: f32) -> f32 {
    let g2 = ga * ga;
    let denom = 1.0 + g2 - 2.0 * ga * mu;
    return (1.0 - g2) / (12.566370614359172 * denom * sqrt(denom));
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covering the viewport.
    var p = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) {
        p = vec2<f32>(3.0, -1.0);
    }
    if (vi == 2u) {
        p = vec2<f32>(-1.0, 3.0);
    }
    return vec4<f32>(p, 0.0, 1.0);
}

@fragment
fn fs_gas(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Pixel center -> NDC (x right, y up), the splat projection's convention.
    let ndc = vec2<f32>(pos.x / u.viewport.x - 1.0, 1.0 - pos.y / u.viewport.y);
    let lateral = u.right.xyz * (ndc.x * u.view.x) + u.up.xyz * (ndc.y * u.view.y);
    var origin: vec3<f32>;
    var dir: vec3<f32>;
    var t_min = -1e30;
    if (u.params.x < 0.5) {
        // Orthographic: parallel rays, the full chord contributes.
        origin = u.view_target.xyz + lateral;
        dir = u.forward.xyz;
    } else {
        // Perspective: eye rays, nothing behind the eye (t >= 0).
        let eye = u.view_target.xyz - u.forward.xyz * u.view.z;
        origin = eye;
        dir = normalize(u.view_target.xyz + lateral - eye);
        t_min = 0.0;
    }

    let tt = clip_aabb(origin, dir, g.mmin.xyz, g.mmax.xyz);
    let t0 = max(tt.x, t_min);
    let t1 = tt.y;
    if (t0 >= t1) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let n = clamp(u32(ceil((t1 - t0) / g.kms.z)), 1u, {max_steps}u);
    let ds = (t1 - t0) / f32(n);

    // Single-scatter starlight: active only with a positive strength AND
    // lights present (the uniform carries count 0 otherwise) — the off path
    // adds nothing and stays bit-identical to the pre-scatter march.
    let n_lights = u32(g.scat.z);
    let scatter_on = g.scat.x > 0.0 && n_lights > 0u;
    let w_out = -dir;

    var t = 1.0;
    var c = vec3<f32>(0.0);
    for (var i = 0u; i < n; i++) {
        let s = t0 + (f32(i) + 0.5) * ds;
        let p = origin + dir * s;
        let rho = density_at(p);
        // Emit THEN attenuate — volume::march_gas's exact operation order.
        let e = t * g.ce.w * rho * ds;
        c += e * g.ce.xyz;
        if (scatter_on) {
            // volume::march_gas's scatter block, operation-for-operation.
            var inc = vec3<f32>(0.0);
            for (var k = 0u; k < n_lights; k++) {
                let dv = p - lights[k].pos;
                let d2_true = dot(dv, dv);
                let d2 = d2_true + lights[k].radius * lights[k].radius;
                if (d2 <= 0.0) {
                    continue;
                }
                var mu = 0.0;
                if (d2_true > 0.0) {
                    mu = dot(dv, w_out) / sqrt(d2_true);
                }
                let f = hg_phase(mu, g.scat.y) / (12.566370614359172 * d2);
                inc += lights[k].rgb * f;
            }
            c += (t * g.scat.x * rho * ds) * inc;
        }
        t = t * exp(-(g.kms.x * rho * ds));
        if (t < {exit}) {
            break;
        }
    }
    return vec4<f32>(c, 1.0 - t);
}
"#;

/// The transmittance prepass: one thread per star marches the mixed density
/// from the star toward the camera and writes `T = exp(−τ)` (τ summed, one
/// exponentiation — volume::star_transmittance's exact order). `{max_steps}`
/// is injected from the `volume` constant; `{workgroup}` from [`PREPASS_WORKGROUP`].
const WGSL_PREPASS: &str = r#"
struct Splat {
    pos: vec3<f32>,
    radius: f32,
    emissive: vec3<f32>,
    pad: f32,
};
@group(2) @binding(0) var<storage, read> splats: array<Splat>;
@group(2) @binding(1) var<storage, read_write> t_out: array<f32>;

@compute @workgroup_size({workgroup})
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&t_out)) {
        return;
    }
    let star = splats[i].pos;
    var dir: vec3<f32>;
    var t_max = 1e30;
    if (u.params.x < 0.5) {
        // Orthographic: toward the camera at infinity, against the view axis.
        dir = -u.forward.xyz;
    } else {
        let eye = u.view_target.xyz - u.forward.xyz * u.view.z;
        let d = eye - star;
        let dist = length(d);
        if (dist == 0.0) {
            t_out[i] = 1.0;
            return;
        }
        dir = d / dist;
        t_max = dist;
    }

    let tt = clip_aabb(star, dir, g.mmin.xyz, g.mmax.xyz);
    let t0 = max(tt.x, 0.0);
    let t1 = min(tt.y, t_max);
    if (t0 >= t1) {
        t_out[i] = 1.0;
        return;
    }
    let n = clamp(u32(ceil((t1 - t0) / g.kms.z)), 1u, {max_steps}u);
    let ds = (t1 - t0) / f32(n);
    var tau = 0.0;
    for (var k = 0u; k < n; k++) {
        let s = t0 + (f32(k) + 0.5) * ds;
        tau += g.kms.x * density_at(star + dir * s) * ds;
    }
    t_out[i] = exp(-tau);
}
"#;

/// Threads per prepass workgroup (one star per thread).
const PREPASS_WORKGROUP: u32 = 64;

/// Unit quad (two triangles) in local [-1, 1] space, scaled per splat by `half`.
const QUAD: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

/// The gas uniform block, mirroring the WGSL `GasUniforms` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GasUniforms {
    /// xyz: emission color tint; w: emissivity `j`.
    color_emissivity: [f32; 4],
    /// x: opacity κ; y: endpoint mix `u`; z: nominal march step; w: unused.
    kappa_mix_step: [f32; 4],
    /// Grid 0 bounds (xyz; w unused).
    b0_min: [f32; 4],
    b0_max: [f32; 4],
    /// Grid 1 bounds (xyz; w unused).
    b1_min: [f32; 4],
    b1_max: [f32; 4],
    /// Union AABB of both grids — the march domain.
    march_min: [f32; 4],
    march_max: [f32; 4],
    /// Single-scatter starlight: x = strength σ_s, y = HG anisotropy g,
    /// z = light count (0 = off, the bit-compat path), w unused.
    scat: [f32; 4],
}

/// One point light as uploaded to the GPU, mirroring the WGSL `PointLight`
/// (and carrying exactly [`crate::volume::Light`]'s fields).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    pos: [f32; 3],
    radius: f32,
    rgb: [f32; 3],
    _pad: f32,
}

/// The reusable GPU rendering context: adapter/device/queue + the splat, gas,
/// and transmittance-prepass pipelines, created once and driven for every
/// frame of a movie.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    gas_pipeline: wgpu::RenderPipeline,
    prepass_pipeline: wgpu::ComputePipeline,
    quad_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    gas_uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Per-frame bind-group layouts: the star pass's transmittance buffer, the
    /// gas uniform + endpoint textures, and the prepass's splat/T-out pair.
    star_t_bgl: wgpu::BindGroupLayout,
    gas_bgl: wgpu::BindGroupLayout,
    prepass_io_bgl: wgpu::BindGroupLayout,
}

impl Renderer {
    /// Bring up a headless wgpu device with the features the renderer needs and
    /// build the splat pipeline. Returns a typed [`RenderError`] (never panics) if
    /// no adapter or required feature is available.
    pub fn new() -> Result<Self, RenderError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // headless
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;

        if !adapter
            .features()
            .contains(wgpu::Features::FLOAT32_BLENDABLE)
        {
            return Err(RenderError::MissingFeature("FLOAT32_BLENDABLE".to_string()));
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("galaxy-render-device"),
                required_features: wgpu::Features::FLOAT32_BLENDABLE,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RenderError::Device(e.to_string()))?;

        let quad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat-quad"),
            contents: bytemuck::cast_slice(&QUAD),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat-uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gas_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gas-uniforms"),
            size: std::mem::size_of::<GasUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat-bind-group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Star pass group 1: the per-star transmittance buffer.
        let star_t_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("star-t-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        // Gas group 1 (gas pass + prepass): gas uniforms + the two endpoint
        // density textures (textureLoad only — no sampler, no FLOAT32_FILTERABLE).
        let tex3d = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        };
        let gas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gas-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: tex3d,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: tex3d,
                    count: None,
                },
                // Scatter point lights: read by the gas fragment march only
                // (the prepass shares this layout but never touches them).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // Prepass group 2: star instances in, transmittance out.
        let prepass_io_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prepass-io-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat-pl"),
            bind_group_layouts: &[Some(&bgl), Some(&star_t_bgl)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{WGSL_UNIFORMS}{SHADER}").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuSplat>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32, 3 => Float32x3],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: ADDITIVE,
                        alpha: ADDITIVE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Gas pass: fullscreen triangle, no vertex buffers, same additive target.
        let exit = format!("{:e}", crate::volume::EXIT_TRANSMITTANCE);
        let max_steps = crate::volume::MAX_STEPS.to_string();
        let gas_src = format!("{WGSL_UNIFORMS}{WGSL_GAS_COMMON}{WGSL_GAS_PASS}")
            .replace("{exit}", &exit)
            .replace("{max_steps}", &max_steps);
        let gas_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gas-shader"),
            source: wgpu::ShaderSource::Wgsl(gas_src.into()),
        });
        let gas_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gas-pl"),
            bind_group_layouts: &[Some(&bgl), Some(&gas_bgl)],
            immediate_size: 0,
        });
        let gas_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gas-pipeline"),
            layout: Some(&gas_layout),
            vertex: wgpu::VertexState {
                module: &gas_shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &gas_shader,
                entry_point: Some("fs_gas"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: ADDITIVE,
                        alpha: ADDITIVE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Transmittance prepass: one thread per star.
        let prepass_src = format!("{WGSL_UNIFORMS}{WGSL_GAS_COMMON}{WGSL_PREPASS}")
            .replace("{max_steps}", &max_steps)
            .replace("{workgroup}", &PREPASS_WORKGROUP.to_string());
        let prepass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prepass-shader"),
            source: wgpu::ShaderSource::Wgsl(prepass_src.into()),
        });
        let prepass_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prepass-pl"),
            bind_group_layouts: &[Some(&bgl), Some(&gas_bgl), Some(&prepass_io_bgl)],
            immediate_size: 0,
        });
        let prepass_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prepass-pipeline"),
            layout: Some(&prepass_layout),
            module: &prepass_shader,
            entry_point: Some("cs_transmittance"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Renderer {
            device,
            queue,
            pipeline,
            gas_pipeline,
            prepass_pipeline,
            quad_buf,
            uniform_buf,
            gas_uniform_buf,
            bind_group,
            star_t_bgl,
            gas_bgl,
            prepass_io_bgl,
        })
    }

    /// Render one frame with an optional volumetric gas component (M7e, plan D9):
    ///
    /// 1. **Transmittance prepass** (compute): one thread per star marches the
    ///    mixed density grid from star to camera and writes `T = exp(−τ)` to a
    ///    storage buffer.
    /// 2. **Star pass**: the splat pipeline, each instance's emission × `T`.
    /// 3. **Gas pass**: a fullscreen triangle raymarches emission+absorption
    ///    per pixel, additively blended into the same `Rgba32Float` target.
    ///
    /// `gas: None` renders stars only, `T ≡ 1.0` — bit-compatible with
    /// [`Renderer::render_frame`] and pinned by the M6g golden gate. The march
    /// rules and their CPU oracles live in [`crate::volume`]; the shaders here
    /// mirror them operation-for-operation.
    pub fn render_frame_with_gas(
        &self,
        frame: &FrameData,
        gas: Option<&crate::volume::GasFrame<'_>>,
        camera: &Camera,
        cfg: &RenderConfig,
    ) -> Result<HdrImage, RenderError> {
        // World-space instances: projection is the vertex shader's job.
        let splats: Vec<GpuSplat> = (0..frame.len())
            .map(|i| {
                let col = frame.color[i];
                let b = frame.brightness[i];
                GpuSplat {
                    pos: frame.pos[i].to_array(),
                    radius: frame.size[i],
                    emissive: [col[0] * b, col[1] * b, col[2] * b],
                    _pad: 0.0,
                }
            })
            .collect();

        let (mode, distance, near) = match camera.projection {
            Projection::Orthographic => (0.0, 0.0, 0.0),
            Projection::Perspective { distance, near } => {
                // The clamp window must be a valid interval in pixels — a
                // min_splat_px above the max would make the WGSL clamp() UB.
                let max_px = cfg.max_splat_ndc * cfg.height as f32 / 2.0;
                let clamps_valid = cfg.min_splat_px.is_finite()
                    && cfg.min_splat_px >= 0.0
                    && cfg.max_splat_ndc.is_finite()
                    && cfg.max_splat_ndc > 0.0
                    && cfg.min_splat_px <= max_px;
                if !clamps_valid {
                    return Err(RenderError::Config(format!(
                        "perspective splat clamps invalid: min_splat_px {} must be finite, \
                         ≥ 0, and ≤ max_splat_ndc·height/2 = {max_px}",
                        cfg.min_splat_px
                    )));
                }
                (1.0, distance, near)
            }
        };
        self.queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&Uniforms {
                right: camera.right.extend(0.0).to_array(),
                up: camera.up.extend(0.0).to_array(),
                forward: camera.forward.extend(0.0).to_array(),
                target: camera.target.extend(0.0).to_array(),
                view: [camera.half_extent.x, camera.half_extent.y, distance, near],
                params: [mode, cfg.falloff, cfg.min_splat_px, cfg.max_splat_ndc],
                viewport: [cfg.width as f32 / 2.0, cfg.height as f32 / 2.0, 0.0, 0.0],
            }),
        );

        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr-accum"),
            size: wgpu::Extent3d {
                width: cfg.width,
                height: cfg.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let instance_buf = (!splats.is_empty()).then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat-instances"),
                    // STORAGE: the transmittance prepass reads star positions
                    // from the same buffer the star pass draws.
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                    contents: bytemuck::cast_slice(&splats),
                })
        });

        // Gas resources: validate the grids, upload the two endpoint density
        // textures, fill the gas uniform block, and build the gas bind group.
        let gas_bg = match gas {
            None => None,
            Some(gf) => {
                let max3d = self.device.limits().max_texture_dimension_3d;
                for (what, g) in [("grid0", gf.grid0), ("grid1", gf.grid1)] {
                    let cells = g.dims.iter().map(|&d| d as usize).product::<usize>();
                    if g.dims.contains(&0) || g.data.len() != cells {
                        return Err(RenderError::Config(format!(
                            "gas {what} holds {} cells but dims {:?} require {cells}",
                            g.data.len(),
                            g.dims
                        )));
                    }
                    if g.dims.iter().any(|&d| d > max3d) {
                        return Err(RenderError::Config(format!(
                            "gas {what} dims {:?} exceed the device 3D texture limit {max3d}",
                            g.dims
                        )));
                    }
                    if !g.bounds_max.cmpgt(g.bounds_min).all() {
                        return Err(RenderError::Config(format!(
                            "gas {what} bounds must have positive extent: {:?}..{:?}",
                            g.bounds_min, g.bounds_max
                        )));
                    }
                }
                // Scatter lights: uploaded only when the look scatters (a
                // positive strength). Empty/off binds one zeroed dummy light
                // with count 0 — the shader's guard never reads it.
                let (strength, anisotropy) = gf
                    .look
                    .scatter
                    .map_or((0.0, 0.0), |s| (s.strength, s.anisotropy));
                let gpu_lights: Vec<GpuLight> = if strength > 0.0 {
                    gf.lights
                        .iter()
                        .map(|l| GpuLight {
                            pos: l.pos.to_array(),
                            radius: l.radius,
                            rgb: l.rgb,
                            _pad: 0.0,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let n_lights = gpu_lights.len() as u32;
                let dummy = GpuLight::zeroed();
                let lights_buf =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("scatter-lights"),
                            contents: if gpu_lights.is_empty() {
                                bytemuck::bytes_of(&dummy)
                            } else {
                                bytemuck::cast_slice(&gpu_lights)
                            },
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                let ext = |v: glam::Vec3| [v.x, v.y, v.z, 0.0];
                let mmin = gf.grid0.bounds_min.min(gf.grid1.bounds_min);
                let mmax = gf.grid0.bounds_max.max(gf.grid1.bounds_max);
                self.queue.write_buffer(
                    &self.gas_uniform_buf,
                    0,
                    bytemuck::bytes_of(&GasUniforms {
                        color_emissivity: [
                            gf.look.color[0],
                            gf.look.color[1],
                            gf.look.color[2],
                            gf.look.emissivity,
                        ],
                        kappa_mix_step: [
                            gf.look.opacity,
                            gf.mix,
                            crate::volume::step_size(gf.grid0, gf.grid1),
                            0.0,
                        ],
                        b0_min: ext(gf.grid0.bounds_min),
                        b0_max: ext(gf.grid0.bounds_max),
                        b1_min: ext(gf.grid1.bounds_min),
                        b1_max: ext(gf.grid1.bounds_max),
                        march_min: ext(mmin),
                        march_max: ext(mmax),
                        scat: [strength, anisotropy, n_lights as f32, 0.0],
                    }),
                );
                let v0 = self.upload_grid(gf.grid0, "gas-rho0");
                let v1 = self.upload_grid(gf.grid1, "gas-rho1");
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gas-bind-group"),
                    layout: &self.gas_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.gas_uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&v0),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&v1),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: lights_buf.as_entire_binding(),
                        },
                    ],
                }))
            }
        };

        // Per-star transmittance: prepass-written when gas is on, constant 1.0
        // when off (×1.0 in the vertex shader is bit-exact — the golden gate).
        let star_t = (!splats.is_empty()).then(|| {
            let buf = if gas_bg.is_some() {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("star-t"),
                    size: (splats.len() * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            } else {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("star-t"),
                        contents: bytemuck::cast_slice(&vec![1.0_f32; splats.len()]),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("star-t-bind-group"),
                layout: &self.star_t_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            (buf, bg)
        });

        // Prepass I/O: star positions in, transmittances out (gas + stars only).
        let prepass_bg = match (&gas_bg, &instance_buf, &star_t) {
            (Some(_), Some(inst), Some((t_buf, _))) => {
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("prepass-io-bind-group"),
                    layout: &self.prepass_io_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: inst.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: t_buf.as_entire_binding(),
                        },
                    ],
                }))
            }
            _ => None,
        };

        // Readback rows must be 256-byte aligned; pad, then strip the padding.
        let unpadded = cfg.width * 16;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * cfg.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // 1. Transmittance prepass: one thread per star, before the star pass.
        if let (Some(gas_bg), Some(io_bg)) = (&gas_bg, &prepass_bg) {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transmittance-prepass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepass_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, gas_bg, &[]);
            pass.set_bind_group(2, io_bg, &[]);
            pass.dispatch_workgroups((splats.len() as u32).div_ceil(PREPASS_WORKGROUP), 1, 1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Clear to fully transparent black — flux starts at zero.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // 2. Star pass: splats × per-star transmittance.
            if let (Some(inst), Some((_, star_bg))) = (&instance_buf, &star_t) {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_bind_group(1, star_bg, &[]);
                pass.set_vertex_buffer(0, self.quad_buf.slice(..));
                pass.set_vertex_buffer(1, inst.slice(..));
                pass.draw(0..6, 0..splats.len() as u32);
            }
            // 3. Gas pass: fullscreen raymarch, additive into the same target
            //    (both terms carry their own attenuation — order-independent).
            if let Some(gas_bg) = &gas_bg {
                pass.set_pipeline(&self.gas_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_bind_group(1, gas_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(cfg.height),
                },
            },
            wgpu::Extent3d {
                width: cfg.width,
                height: cfg.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);

        // Map, block until the GPU is done, and un-pad into row-major RGBA.
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::BufferMap(e.to_string()))?;
        rx.recv()
            .map_err(|e| RenderError::BufferMap(e.to_string()))?
            .map_err(|e| RenderError::BufferMap(e.to_string()))?;

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((cfg.width * cfg.height) as usize);
        for y in 0..cfg.height {
            let row_start = (y * padded) as usize;
            let row = &data[row_start..row_start + unpadded as usize];
            let floats: &[f32] = bytemuck::cast_slice(row);
            for x in 0..cfg.width {
                let i = (x * 4) as usize;
                pixels.push([floats[i], floats[i + 1], floats[i + 2], floats[i + 3]]);
            }
        }
        drop(data);
        readback.unmap();

        Ok(HdrImage {
            width: cfg.width,
            height: cfg.height,
            pixels,
        })
    }

    /// Render one frame: additively blend every particle in `frame` as a Gaussian
    /// splat, as seen by `camera`, into an `Rgba32Float` target of
    /// `cfg.width × cfg.height`, and read it back as a linear [`HdrImage`].
    /// Equivalent to [`Renderer::render_frame_with_gas`] with no gas.
    pub fn render_frame(
        &self,
        frame: &FrameData,
        camera: &Camera,
        cfg: &RenderConfig,
    ) -> Result<HdrImage, RenderError> {
        self.render_frame_with_gas(frame, None, camera, cfg)
    }

    /// Upload one gas density grid as an `R32Float` 3D texture and return its
    /// view (the bind group keeps the texture alive). `data` is x-fastest,
    /// exactly the texel order `write_texture` consumes.
    fn upload_grid(&self, grid: &galaxy_renderprep::GasGrid, label: &str) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: grid.dims[0],
            height: grid.dims[1],
            depth_or_array_layers: grid.dims[2],
        };
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&grid.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(grid.dims[0] * 4),
                rows_per_image: Some(grid.dims[1]),
            },
            size,
        );
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

/// Additive blend factor pair: `dst = src·1 + dst·1` (order-independent accumulation).
const ADDITIVE: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::One,
    dst_factor: wgpu::BlendFactor::One,
    operation: wgpu::BlendOperation::Add,
};
