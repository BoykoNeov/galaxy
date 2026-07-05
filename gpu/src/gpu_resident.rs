//! [`GpuResidentLeapfrog`]: GPU-**resident** leapfrog stepping (DESIGN M4i) — keeping particle
//! *state* on the GPU across integrator steps, the payoff M4h's single-device fuse unlocked.
//!
//! ## What residency buys, and what it costs
//! [`crate::GpuLbvhFused`] (M4h) fused the whole LBVH build+traverse onto one device in one
//! submit, but still **uploads state and reads back accel every `accelerations` call** — one
//! CPU↔GPU round-trip per force evaluation. `GpuResidentLeapfrog` closes that loop: `pos`, `vel`,
//! `mass`, and `acc` live in GPU storage buffers *across* steps, the kick/drift arithmetic runs
//! on the device, and nothing crosses the bus until an explicit [`snapshot`](Self::snapshot).
//! `bodies` (xyz=pos, w=mass) — already the traversal's input — doubles as the resident position
//! buffer, so the force pipeline reads state in place.
//!
//! Residency is *not* a throughput speedup (the M4h serial stages — sort, aggregate, flatten — are
//! unchanged and still dominate); it removes the per-step sync points, the point of residency.
//! **Batching** then drops the per-*submit* overhead on top (M4k): [`step`](Self::step) is the
//! one-submit minimum-latency path, while [`step_many`](Self::step_many) coalesces up to
//! [`MAX_BATCH`](Self::MAX_BATCH) steps into a single encoder/submit — regrouping encoders without
//! touching the arithmetic, so the trajectory is bit-identical to stepping one at a time.
//!
//! ## Position precision: double-single accumulation (M4j)
//! The host-driven path ([`galaxy_core::LeapfrogKdk`] + a solver) keeps **authoritative
//! positions in f64** and re-narrows to f32 only to feed the GPU force kernel each step. WGSL has
//! no portable `f64`, so the resident path instead carries positions as a **double-single**
//! (`hi + lo`, an unevaluated pair of f32s ≈ 46-bit mantissa): the [drift kernel](DRIFT_SHADER)
//! accumulates `pos += vel*dt` with a compensated two-sum, so the small per-step increment is no
//! longer lost into the growing coordinate's f32 ulp. `hi` is `bodies.xyz` — the force pipeline
//! still reads only that f32, so build/traverse and their gates are untouched; the force *itself*
//! stays f32 (mirroring the existing f32-force / f64-energy note). `upload` splits the f64 input
//! into `hi + lo` and `snapshot` sums them back, so the full f64 precision reaches the host. This
//! is the M4i precision follow-up DESIGN deferred; velocity remains plain f32 (DS is
//! position-only). The remaining M4i throughput follow-up — **batching K steps into one submit** —
//! landed as M4k (see [`step_many`](Self::step_many)).
//!
//! ## Not a `ForceSolver`
//! The [`galaxy_core::ForceSolver`] interface is host-state-in / accel-out — fundamentally
//! incompatible with keeping state resident. So this is its own type with an
//! `upload → step* → snapshot` lifecycle, exactly as DESIGN "Remaining M4+" anticipated.

use bytemuck::{Pod, Zeroable};

use galaxy_core::{DVec3, Species, State};
use galaxy_solvers::sph::{DensityConfig, HydroParams};

use crate::fused_core::{bg_entry, storage_entry, uniform_entry, FusedCore};
use crate::GpuError;

/// Workgroup width for the per-particle kick/drift/reset kernels.
const WG: u32 = 256;

/// `NO_PARENT` = `u32::MAX`, the root sentinel the aggregate walk stops on. The reset kernel
/// re-writes every `parent` slot to this each force evaluation (see [`galaxy_solvers::NO_PARENT`]).
const NO_PARENT_LIT: &str = "4294967295u";

/// Re-seed `idx_a` (iota) and `parent` (`NO_PARENT`) on the GPU each force evaluation — the
/// on-device equivalent of the fused solver's per-call host `write_buffer`s, so no state leaves
/// the device between steps. `parent` needs the `NO_PARENT` pre-fill because the Karras build only
/// writes children's parent slots (never the root's), and the aggregate walk stops on it.
fn reset_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, dt: f32, half_dt: f32, pad: u32 }};
@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> idx_a:  array<u32>;
@group(0) @binding(2) var<storage, read_write> parent: array<u32>;

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let k = gid.x;
    let total = 2u * params.n - 1u;
    if (k >= total) {{ return; }}
    if (k < params.n) {{ idx_a[k] = k; }}
    parent[k] = {NO_PARENT_LIT};
}}
"#
    )
}

/// Leapfrog half-kick: `vel.xyz += accel.xyz * half_dt`, preserving `vel.w`.
const KICK_SHADER: &str = r#"
struct Params { n: u32, dt: f32, half_dt: f32, pad: u32 };
@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> vel:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       accel:  array<vec4<f32>>;

@compute @workgroup_size(256)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let v = vel[i];
    vel[i] = vec4<f32>(v.xyz + accel[i].xyz * params.half_dt, v.w);
}
"#;

/// Leapfrog drift with **double-single** (float-float) position accumulation: `pos += vel*dt`
/// carried as an unevaluated `hi + lo` f32 pair, so the small per-step increment is not lost into
/// the growing coordinate's ulp (the f32-accumulation precision cost DESIGN M4i deferred). `hi`
/// lives in `bodies.xyz` (also the force pipeline's input — forces read only the f32 `hi`, so
/// build/traverse are untouched); `lo` lives in the resident `pos_lo.xyz`. `bodies.w` (= mass) is
/// preserved.
///
/// Each step folds the single-f32 increment `d = vel*dt` into `(hi, lo)` with a compensated add:
/// Knuth `two_sum(hi, d) → (s, err)` recovers the rounding error `hi + d` loses, `err += lo` folds
/// in the carried low part, then `quick_two_sum(s, err) → (hi', lo')` renormalizes so `|lo'| ≤
/// ½ulp(hi')`. That normalization is load-bearing: it makes the f64 snapshot↔upload round-trip a
/// bit-exact identity, which is what keeps the M4i faithful/residency gate exact.
///
/// ## Defeating f32 reassociation (the `ax`/bitcast barriers)
/// Both error-free transforms rely on IEEE non-associativity, and consumer-GPU f32 compilers
/// reassociate by default — which collapses the compensation to *exactly zero* (measured on the
/// Vulkan test adapter: without barriers the DS result is bit-identical to a plain-f32 running
/// sum). Two folds do the damage: `two_sum`'s `s - hi → d` (value-dependent) and `s - (s - hi) →
/// hi` **and** `quick_two_sum`'s `(s + e) - s → e` (both value-*independent* identities that hold
/// for any operand). `ax(x) = bitcast<f32>(bitcast<u32>(x))` is a value-preserving round-trip that
/// is opaque to the FP optimizer, forcing the true IEEE-rounded intermediate. `s`, `bb` and
/// `hi_new` must **all** be laundered — laundering only `s` leaves the value-independent folds
/// intact and the error term still vanishes. `d` is also isolated in its own `let` so no `a*b+c`
/// remains to contract into an fma. **Caveat:** GPU emulated-double is driver-dependent; the M4j
/// gate proves this on the test adapter, not universally.
const DRIFT_SHADER: &str = r#"
struct Params { n: u32, dt: f32, half_dt: f32, barrier: u32 };
@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> bodies: array<vec4<f32>>; // xyz=pos hi, w=mass
@group(0) @binding(2) var<storage, read>       vel:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pos_lo: array<vec4<f32>>; // xyz=pos lo

// Value-preserving optimization barrier: XOR the bits with `params.barrier`, a uniform the host
// pins to 0. Because it is a runtime uniform (not a compile-time constant), the compiler cannot
// prove the XOR is identity, so it cannot fold `bitcast(bitcast(x) ^ barrier)` back to `x` — this
// forces the real IEEE-rounded value and blocks the additive reassociation that would otherwise
// collapse the two-sum. (A plain `bitcast<f32>(bitcast<u32>(x))` round-trip was *not* enough:
// naga folds it away, and the DS result stayed bit-identical to a plain-f32 sum on the Vulkan
// adapter.) `barrier` occupies the params slot the kick/drift/reset uniform already carried as pad.
fn ax(v: vec3<f32>, barrier: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(v) ^ vec3<u32>(barrier));
}

@compute @workgroup_size(256)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let m = params.barrier;
    let b = bodies[i];
    let hi = b.xyz;
    let lo = pos_lo[i].xyz;
    let d = vel[i].xyz * params.dt;            // isolated f32 increment (no fma to contract)
    // Knuth two_sum(hi, d) -> (s, err): err is the exact rounding error of hi + d. s and bb are
    // laundered so neither `s - hi` nor `s - bb` reassociates to a closed form.
    let s = ax(hi + d, m);
    let bb = ax(s - hi, m);
    let err = (hi - (s - bb)) + (d - bb);
    // Fold the carried low part, then renormalize via quick_two_sum(s, e) (|s| >= |e|). hi_new is
    // laundered so `hi_new - s` does not reassociate to `e`.
    let e = err + lo;
    let hi_new = ax(s + e, m);
    let lo_new = e - (hi_new - s);
    bodies[i] = vec4<f32>(hi_new, b.w);
    pos_lo[i] = vec4<f32>(lo_new, 0.0);
}
"#;

/// Uniform for the kick/drift/reset kernels: particle count + this step's `dt` / `half_dt`.
/// `reset` reads only `.n`; `kick` reads `.half_dt`; `drift` reads `.dt` and `.barrier`. `barrier`
/// is pinned to 0 and used only by drift's double-single `ax` optimization barrier (an XOR mask
/// the compiler can't fold because it's a runtime uniform); it reuses the old padding slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepParams {
    n: u32,
    dt: f32,
    half_dt: f32,
    barrier: u32,
}

/// Resident-owned resources (the resident velocity + double-single position low-part buffers, the
/// readback buffers, and the kick/drift/reset bind groups). Rebuilt when [`FusedCore`] grows. The
/// position *high* part is [`FusedCore`]'s `bodies` (shared with the force pipeline); `pos_lo` is
/// resident-only and holds the low half of each coordinate's `hi + lo` double-single.
struct ResidentResources {
    vel: wgpu::Buffer,
    pos_lo: wgpu::Buffer,
    pos_readback: wgpu::Buffer,
    pos_lo_readback: wgpu::Buffer,
    vel_readback: wgpu::Buffer,
    reset_bg: wgpu::BindGroup,
    kick_bg: wgpu::BindGroup,
    drift_bg: wgpu::BindGroup,
    capacity: usize,
}

/// Gather the gas subset off the resident `bodies` buffer into the compact,
/// interleaved-f32 layout the SPH density/hydro kernels expect. `bodies.xyz` is the
/// double-single *hi* limb (the same f32 position the gravity force reads — D1: SPH is
/// f32 anyway); `bodies.w` is mass. One invocation per gas particle; unique gas indices
/// ⇒ no scatter race. This is the resident analogue of the CPU composite's
/// `gas.iter().map(|&i| state.pos[i])` compaction.
const GATHER_GAS_SHADER: &str = r#"
struct GParams { n_gas: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform>             gp:       GParams;
@group(0) @binding(1) var<storage, read>       bodies:   array<vec4<f32>>; // xyz=pos hi, w=mass
@group(0) @binding(2) var<storage, read>       gas_idx:  array<u32>;
@group(0) @binding(3) var<storage, read_write> gas_pos:  array<f32>;       // 3*n_gas interleaved
@group(0) @binding(4) var<storage, read_write> gas_mass: array<f32>;       // n_gas

@compute @workgroup_size(256)
fn gather_gas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= gp.n_gas) { return; }
    let b = bodies[gas_idx[k]];
    gas_pos[3u * k]      = b.x;
    gas_pos[3u * k + 1u] = b.y;
    gas_pos[3u * k + 2u] = b.z;
    gas_mass[k]          = b.w;
}
"#;

/// Uniform for the gas-gather kernel (just the gas count; padded to 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherParams {
    n_gas: u32,
    _pad: [u32; 3],
}

/// Resident SPH per-upload resources. Only the buffers a later pass re-touches are kept:
/// `rho`/`h` (the density outputs, copied out by [`Sph::snapshot`]), their readbacks, and
/// the two bind groups (which run in [`Sph::encode`]). The gather/grid intermediates
/// (`gas_idx`/`gas_pos`/`gas_mass`/`slot_count`/`cursor`/`cell_start`/`sorted_idx`) are
/// **not** stored — each is bound into a bind group above, and a wgpu bind group retains
/// its resources for its own lifetime, so they live exactly as long as these bind groups
/// (the same idiom as [`crate::fused_core`]'s `FusedResources`).
struct SphResources {
    rho: wgpu::Buffer,
    h: wgpu::Buffer,
    rho_readback: wgpu::Buffer,
    h_readback: wgpu::Buffer,
    gather_bg: wgpu::BindGroup,
    density_bg: wgpu::BindGroup,
}

/// Resident isothermal-SPH state carried by a stepper in **gas mode** (via
/// [`GpuResidentLeapfrog::new_with_sph`]). Holds the config, the gather + density (build /
/// adaptive-h root-find) pipelines built on [`FusedCore`]'s device, the two fixed-size
/// uniform buffers (written at upload), and the lazily-sized [`SphResources`]. The
/// resident analogue of the CPU composite [`galaxy_solvers::sph::GravitySph`] — minus the
/// `h_hint` warm-start, which the seed-independent GPU root-find does not need (G2). The
/// hydro-force stage lands in G5b; G5a builds density only.
struct Sph {
    #[allow(dead_code)] // consumed by the G5b hydro-force stage
    hydro: HydroParams,
    density: DensityConfig,
    gather_pl: wgpu::ComputePipeline,
    build_pl: wgpu::ComputePipeline,
    density_pl: wgpu::ComputePipeline,
    gather_bgl: wgpu::BindGroupLayout,
    density_bgl: wgpu::BindGroupLayout,
    gather_params_buf: wgpu::Buffer,
    density_params_buf: wgpu::Buffer,
    res: Option<SphResources>,
}

/// Map a resident f32 readback buffer, block, and copy out the first `count` scalars.
fn map_read_f32(device: &wgpu::Device, readback: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count * std::mem::size_of::<f32>()) as u64;
    let slice = readback.slice(..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("gpu poll failed");
    rx.recv()
        .expect("map channel closed")
        .expect("gpu buffer map failed");
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice::<u8, f32>(&data)[..count].to_vec();
    drop(data);
    readback.unmap();
    out
}

impl Sph {
    /// Build the gather + density (build / adaptive-h) pipelines on the resident
    /// [`FusedCore`] device. Reuses the G2 density WGSL verbatim
    /// (`DENSITY_DECLS + GRID_HELPERS_WGSL + DENSITY_KERNELS`) so the root-find is one
    /// source of truth with the standalone [`crate::GpuDensity`].
    fn new(device: &wgpu::Device, hydro: HydroParams, density: DensityConfig) -> Self {
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let gather_mod = module("resident-sph-gather", GATHER_GAS_SHADER);
        let density_src = format!(
            "{}{}{}",
            crate::sph_density::DENSITY_DECLS,
            crate::sph_grid::GRID_HELPERS_WGSL,
            crate::sph_density::DENSITY_KERNELS,
        );
        let density_mod = module("resident-sph-density", &density_src);

        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        // gather: 0 uniform, 1 bodies(r), 2 gas_idx(r), 3 gas_pos(rw), 4 gas_mass(rw)
        let gather_bgl = bgl(
            "resident-sph-gather-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        );
        // density (build + root-find share it): 0 uniform, 1 pos(r), 2 mass(r),
        // 3 slot_count(rw), 4 cursor(rw), 5 cell_start(rw), 6 sorted_idx(rw), 7 h_io(rw),
        // 8 rho_out(rw) — matches DENSITY_DECLS.
        let density_bgl = bgl(
            "resident-sph-density-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        );

        let pipeline = |label: &str,
                        layout: &wgpu::BindGroupLayout,
                        module: &wgpu::ShaderModule,
                        entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let gather_pl = pipeline(
            "resident-sph-gather",
            &gather_bgl,
            &gather_mod,
            "gather_gas",
        );
        let build_pl = pipeline("resident-sph-build", &density_bgl, &density_mod, "build");
        let density_pl = pipeline(
            "resident-sph-density",
            &density_bgl,
            &density_mod,
            "density_adaptive",
        );

        let uniform_buf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let gather_params_buf = uniform_buf(
            "resident-sph-gather-params",
            std::mem::size_of::<GatherParams>() as u64,
        );
        let density_params_buf = uniform_buf(
            "resident-sph-density-params",
            std::mem::size_of::<crate::sph_density::Params>() as u64,
        );

        Sph {
            hydro,
            density,
            gather_pl,
            build_pl,
            density_pl,
            gather_bgl,
            density_bgl,
            gather_params_buf,
            density_params_buf,
            res: None,
        }
    }

    /// (Re)allocate the SPH buffers, write both uniforms + the gas map, and rebuild the
    /// gather/density bind groups referencing the current `bodies`. Called from
    /// [`GpuResidentLeapfrog::upload`] in gas mode with `gas_idx` non-empty.
    ///
    /// Allocated fresh every upload (not lazily grown): the gather bind group references
    /// `bodies`, which can be reallocated independently of the gas count when the total N
    /// grows, so a cached bind group could point at a stale buffer. Upload is rare (once
    /// per run / per re-IC), so a per-upload realloc of ~10 small buffers is negligible —
    /// the per-*step* SPH path touches none of this.
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bodies: &wgpu::Buffer,
        gas_idx: &[usize],
        gas_pos_host: &[DVec3],
    ) {
        let n_gas = gas_idx.len();
        let table_size = crate::sph_density::table_size_for(n_gas);

        queue.write_buffer(
            &self.gather_params_buf,
            0,
            bytemuck::bytes_of(&GatherParams {
                n_gas: n_gas as u32,
                _pad: [0; 3],
            }),
        );
        // Density seed params (h_seed / h_cap / grid cell) fixed at upload from the initial
        // gas positions. Two distinct staleness stories, only the first fully settled:
        //   • h_seed / h_cap staleness is BENIGN — the bracket expands 64 doublings / 60
        //     halvings, so a stale seed only costs iterations, never the root (seed-
        //     independence, G2). Validated here at/near the primed config.
        //   • The grid `cell` is a residency-specific artifact: the standalone GpuDensity
        //     recomputes it each call to keep the median walk span ~2–3, so freezing it lets
        //     the span drift as the global gas scale evolves. Merger CONTRACTION (the dominant
        //     direction) makes the frozen cell stale-large ⇒ smaller span ⇒ the safe direction;
        //     uniform EXPANSION grows the span toward MAX_SPAN and then clips (undercount). The
        //     wide-h tail clips in BOTH paths (that part is shared D4, not this artifact).
        // Correct at/near the primed config (gated in G5a); per-step behavior over a stepped
        // run is gated in G5b — if drift shows, an on-GPU gas-bbox reduction recomputes `cell`
        // (the same reduction G5c's no-readback CFL min needs).
        let dparams = crate::sph_density::density_params(
            gas_pos_host,
            self.density.n_ngb,
            self.density.h_tol_rel,
        );
        queue.write_buffer(&self.density_params_buf, 0, bytemuck::bytes_of(&dparams));

        let store = wgpu::BufferUsages::STORAGE;
        let cdst = wgpu::BufferUsages::COPY_DST;
        let csrc = wgpu::BufferUsages::COPY_SRC;
        let mapread = wgpu::BufferUsages::MAP_READ;
        let mk = |label: &str, count: usize, elem: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (count * elem) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let f32sz = std::mem::size_of::<f32>();
        let u32sz = std::mem::size_of::<u32>();
        let ts = table_size as usize;
        let gas_idx_buf = mk("resident-sph-gas-idx", n_gas, u32sz, store | cdst);
        let gas_pos = mk("resident-sph-gas-pos", 3 * n_gas, f32sz, store);
        let gas_mass = mk("resident-sph-gas-mass", n_gas, f32sz, store);
        let slot_count = mk("resident-sph-slot-count", ts, u32sz, store);
        let cursor = mk("resident-sph-cursor", ts, u32sz, store);
        let cell_start = mk("resident-sph-cell-start", ts + 1, u32sz, store);
        let sorted_idx = mk("resident-sph-sorted-idx", n_gas, u32sz, store);
        let rho = mk("resident-sph-rho", n_gas, f32sz, store | csrc);
        let h = mk("resident-sph-h", n_gas, f32sz, store | csrc);
        let rho_readback = mk("resident-sph-rho-readback", n_gas, f32sz, cdst | mapread);
        let h_readback = mk("resident-sph-h-readback", n_gas, f32sz, cdst | mapread);

        let gi: Vec<u32> = gas_idx.iter().map(|&i| i as u32).collect();
        queue.write_buffer(&gas_idx_buf, 0, bytemuck::cast_slice(&gi));

        let gather_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident-sph-gather-bg"),
            layout: &self.gather_bgl,
            entries: &[
                bg_entry(0, &self.gather_params_buf),
                bg_entry(1, bodies),
                bg_entry(2, &gas_idx_buf),
                bg_entry(3, &gas_pos),
                bg_entry(4, &gas_mass),
            ],
        });
        let density_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident-sph-density-bg"),
            layout: &self.density_bgl,
            entries: &[
                bg_entry(0, &self.density_params_buf),
                bg_entry(1, &gas_pos),
                bg_entry(2, &gas_mass),
                bg_entry(3, &slot_count),
                bg_entry(4, &cursor),
                bg_entry(5, &cell_start),
                bg_entry(6, &sorted_idx),
                bg_entry(7, &h),
                bg_entry(8, &rho),
            ],
        });

        self.res = Some(SphResources {
            rho,
            h,
            rho_readback,
            h_readback,
            gather_bg,
            density_bg,
        });
    }

    /// Append the resident SPH density stages onto `enc`: gather gas off `bodies`, build
    /// the gas grid, root-find (ρ, h). Left resident for the hydro force (G5b). Each stage
    /// is its own compute pass, so wgpu inserts the read-after-write barriers between them.
    fn encode(&self, enc: &mut wgpu::CommandEncoder, n_gas: usize) {
        let res = self
            .res
            .as_ref()
            .expect("sph resources prepared before encode");
        let pass = |enc: &mut wgpu::CommandEncoder,
                    label: &str,
                    pipeline: &wgpu::ComputePipeline,
                    bg: &wgpu::BindGroup,
                    groups: u32| {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(groups, 1, 1);
        };
        let wide = (n_gas as u32).div_ceil(WG);
        pass(
            enc,
            "resident-sph-gather",
            &self.gather_pl,
            &res.gather_bg,
            wide,
        );
        pass(
            enc,
            "resident-sph-build",
            &self.build_pl,
            &res.density_bg,
            1,
        );
        pass(
            enc,
            "resident-sph-density",
            &self.density_pl,
            &res.density_bg,
            wide,
        );
    }

    /// Copy resident gas (ρ, h) to the host. Caller submits nothing else in between; the
    /// last force evaluation left them resident.
    fn snapshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n_gas: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let res = self
            .res
            .as_ref()
            .expect("sph resources prepared before snapshot");
        let bytes = (n_gas * std::mem::size_of::<f32>()) as u64;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident-sph-density-readback"),
        });
        enc.copy_buffer_to_buffer(&res.rho, 0, &res.rho_readback, 0, bytes);
        enc.copy_buffer_to_buffer(&res.h, 0, &res.h_readback, 0, bytes);
        queue.submit([enc.finish()]);
        let rho = map_read_f32(device, &res.rho_readback, n_gas);
        let h = map_read_f32(device, &res.h_readback, n_gas);
        (rho, h)
    }
}

/// Gas-subset density readback (the G5a gate surface). The resident stepper leaves gas
/// (ρ, h) on the device across steps; [`GpuResidentLeapfrog::snapshot_gas_density`]
/// copies them back paired with the gas map for oracle comparison. `f32` because the
/// device computes in f32 (D1 — the gate is an f32-tolerance comparison, never bit-exact).
pub struct GasDensity {
    /// Global particle indices of the gas rows, ascending — the resident gas map
    /// (`kind == Species::Gas`, matching the CPU composite's gas order).
    pub gas_idx: Vec<usize>,
    /// Per-gas density `ρ_i`.
    pub rho: Vec<f32>,
    /// Per-gas adaptive smoothing length `h_i`.
    pub h: Vec<f32>,
}

/// GPU-resident kick-drift-kick leapfrog over the M4h fused LBVH force pipeline. State stays in
/// GPU buffers across [`step`](Self::step)s; only [`snapshot`](Self::snapshot) reads it back.
///
/// With [`new_with_sph`](Self::new_with_sph) the stepper additionally runs isothermal SPH on
/// the **gas subset** (`kind == Species::Gas`) each force evaluation — gravity over all
/// particles, hydro added to the gas rows — the resident analogue of
/// [`galaxy_solvers::sph::GravitySph`] (GPU-SPH G5).
pub struct GpuResidentLeapfrog {
    core: FusedCore,
    // kick/drift/reset pipelines + layouts (built once).
    reset_pl: wgpu::ComputePipeline,
    kick_pl: wgpu::ComputePipeline,
    drift_pl: wgpu::ComputePipeline,
    reset_bgl: wgpu::BindGroupLayout,
    kick_bgl: wgpu::BindGroupLayout,
    drift_bgl: wgpu::BindGroupLayout,
    step_params_buf: wgpu::Buffer,
    res: Option<ResidentResources>,
    // SPH gas mode (None = gravity-only). `gas_idx` is the ascending-index gas map, rebuilt
    // from `state.kind` on each [`upload`](Self::upload); empty in gravity-only mode.
    sph: Option<Sph>,
    gas_idx: Vec<usize>,
    // Host-tracked bookkeeping.
    n: usize,
    time: f64,
    mass: Vec<f64>,
    // Count of `queue.submit`s issued over this stepper's life (prime + steps + snapshots). The
    // batching gate reads this as a before/after delta to prove `step_many` coalesces submits.
    submits: u64,
}

impl GpuResidentLeapfrog {
    /// Max resident KDK steps [`step_many`](Self::step_many) encodes into a **single submit**
    /// before flushing. Batching drops per-submit overhead (the named M4i throughput follow-up —
    /// M4i removed the per-step *latency*, this removes the per-step *submit*), but a submit that
    /// runs too long trips the OS GPU watchdog (Windows TDR / the Vulkan device-loss timeout the
    /// M4j path is verified on), so the batch is capped rather than unbounded. 64 already collapses
    /// the K=10⁴ drift gate from 10⁴ submits to ~157 (a 64× overhead drop); returns diminish past
    /// that, so it stays conservative.
    ///
    /// **Fixed, not per-N (M4l).** M4i flagged that a *fixed* cap is N-blind — per-step GPU cost
    /// rises with N, so in principle a large-N sim could approach the watchdog at a cap safe for the
    /// small-N gates. The `bench_step_cost` timing bench measured this (RTX 5090 / Vulkan): the
    /// resident step is **overhead-bound** (its ~10 serial LBVH dispatches dominate, not N-scaling
    /// compute), so a full 64-step submit stays ≥10× under the watchdog budget through 1M particles.
    /// Fixed 64 is therefore measured-safe to ≥1M; per-N adaptive sizing is deferred until a real
    /// 10⁷–10⁸ crossover measurement, where the *measured* crossover — not an `n·log n` guess — sets
    /// the knee.
    pub const MAX_BATCH: u64 = 64;

    /// Bring up the resident compute device + every pipeline (the shared [`FusedCore`] build/
    /// traverse plus the new kick/drift/reset kernels). Returns a typed [`GpuError`] on adapter/
    /// device failure.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        let core = FusedCore::new(g, softening, theta)?;
        let dev = &core.device;

        let module = |label: &str, src: &str| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let reset_mod = module("resident-reset", &reset_shader());
        let kick_mod = module("resident-kick", KICK_SHADER);
        let drift_mod = module("resident-drift", DRIFT_SHADER);

        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        // reset: 0 uniform, 1 idx_a(rw), 2 parent(rw)
        let reset_bgl = bgl(
            "resident-reset-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        );
        // kick: 0 uniform, 1 vel(rw), 2 accel(r)
        let kick_bgl = bgl(
            "resident-kick-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        );
        // drift: 0 uniform, 1 bodies/pos-hi(rw), 2 vel(r), 3 pos_lo(rw)
        let drift_bgl = bgl(
            "resident-drift-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        );

        let pipeline = |label: &str,
                        layout: &wgpu::BindGroupLayout,
                        module: &wgpu::ShaderModule,
                        entry: &str| {
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let reset_pl = pipeline("resident-reset", &reset_bgl, &reset_mod, "reset");
        let kick_pl = pipeline("resident-kick", &kick_bgl, &kick_mod, "kick");
        let drift_pl = pipeline("resident-drift", &drift_bgl, &drift_mod, "drift");

        let step_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident-step-params"),
            size: std::mem::size_of::<StepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuResidentLeapfrog {
            core,
            reset_pl,
            kick_pl,
            drift_pl,
            reset_bgl,
            kick_bgl,
            drift_bgl,
            step_params_buf,
            res: None,
            sph: None,
            gas_idx: Vec::new(),
            n: 0,
            time: 0.0,
            mass: Vec::new(),
            submits: 0,
        })
    }

    /// Bring up a resident stepper in **gas mode** (GPU-SPH G5): the shared LBVH gravity
    /// pipeline plus resident isothermal SPH on the gas subset. `hydro`/`density` mirror
    /// the CPU composite [`galaxy_solvers::sph::GravitySph`]'s parameters. Gravity acts on
    /// ALL particles; hydro (and its density prerequisite) on `kind == Species::Gas` only.
    pub fn new_with_sph(
        g: f64,
        softening: f64,
        theta: f64,
        hydro: HydroParams,
        density: DensityConfig,
    ) -> Result<Self, GpuError> {
        let mut s = Self::new(g, softening, theta)?;
        s.sph = Some(Sph::new(&s.core.device, hydro, density));
        Ok(s)
    }

    /// (Re)allocate the resident velocity + readback buffers and rebuild the kick/drift/reset bind
    /// groups (which reference [`FusedCore`]'s `bodies`/`accel`/`idx_a`/`parent`). Called after
    /// `core.ensure_capacity`, so it sees the current core buffers. `cap >= 2` — allocated even for
    /// a single particle so no intermediate buffer is zero-sized.
    fn ensure_capacity(&mut self, cap: usize) {
        if let Some(res) = &self.res {
            if cap <= res.capacity {
                return;
            }
        }
        let dev = &self.core.device;
        let f4 = |count: usize| (count * std::mem::size_of::<[f32; 4]>()) as u64;
        let store = wgpu::BufferUsages::STORAGE;
        let cdst = wgpu::BufferUsages::COPY_DST;
        let csrc = wgpu::BufferUsages::COPY_SRC;
        let mapread = wgpu::BufferUsages::MAP_READ;

        let make_store = |label: &str| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: f4(cap),
                usage: store | cdst | csrc,
                mapped_at_creation: false,
            })
        };
        let vel = make_store("resident-vel");
        // Double-single low part: seeded on upload, mutated by drift, read back on snapshot.
        let pos_lo = make_store("resident-pos-lo");
        let make_rb = |label: &str| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: f4(cap),
                usage: cdst | mapread,
                mapped_at_creation: false,
            })
        };
        let pos_readback = make_rb("resident-pos-readback");
        let pos_lo_readback = make_rb("resident-pos-lo-readback");
        let vel_readback = make_rb("resident-vel-readback");

        let core_res = self.core.res.as_ref().expect("core capacity ensured first");
        let bind =
            |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
                dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries,
                })
            };
        let reset_bg = bind(
            "resident-reset-bg",
            &self.reset_bgl,
            &[
                bg_entry(0, &self.step_params_buf),
                bg_entry(1, &core_res.idx_a),
                bg_entry(2, &core_res.parent),
            ],
        );
        let kick_bg = bind(
            "resident-kick-bg",
            &self.kick_bgl,
            &[
                bg_entry(0, &self.step_params_buf),
                bg_entry(1, &vel),
                bg_entry(2, &core_res.accel),
            ],
        );
        let drift_bg = bind(
            "resident-drift-bg",
            &self.drift_bgl,
            &[
                bg_entry(0, &self.step_params_buf),
                bg_entry(1, &core_res.bodies),
                bg_entry(2, &vel),
                bg_entry(3, &pos_lo),
            ],
        );

        self.res = Some(ResidentResources {
            vel,
            pos_lo,
            pos_readback,
            pos_lo_readback,
            vel_readback,
            reset_bg,
            kick_bg,
            drift_bg,
            capacity: cap,
        });
    }

    /// Write the per-step uniform (`n`, `dt`, `half_dt`). `dt = 0` for the initial prime (only
    /// `.n` is read there).
    fn write_step_params(&self, dt: f64) {
        self.core.queue.write_buffer(
            &self.step_params_buf,
            0,
            bytemuck::bytes_of(&StepParams {
                n: self.n as u32,
                dt: dt as f32,
                half_dt: (0.5 * dt) as f32,
                barrier: 0, // pinned; the drift barrier XORs by this, so it must stay 0
            }),
        );
    }

    /// A one-invocation-per-particle compute pass over the given pipeline + bind group.
    fn per_particle_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
    ) {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups((self.n as u32).div_ceil(WG), 1, 1);
    }

    /// Append the force evaluation into `accel`: for `n >= 2` the reset (re-seed idx_a/parent)
    /// followed by the shared build+traverse; for `n == 1` just zero `accel` (a lone particle
    /// feels no force). Assumes `bodies` holds the current positions.
    fn encode_force(&self, enc: &mut wgpu::CommandEncoder) {
        let res = self.res.as_ref().expect("resident resources ensured");
        if self.n >= 2 {
            self.per_particle_pass(enc, "resident-reset", &self.reset_pl, &res.reset_bg);
            self.core.encode_build_traverse(enc, self.n);
        } else {
            // n == 1: no tree; the single particle's acceleration is exactly zero.
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            enc.clear_buffer(&core_res.accel, 0, None);
        }
        // Gas mode: add the resident SPH force onto the gas rows of `accel` (density
        // root-find → hydro → scatter-add). Encoded AFTER gravity-traverse so the RMW on
        // `accel` sees the gravity contribution (wgpu barriers honor encode order).
        if self.sph.is_some() && !self.gas_idx.is_empty() {
            self.encode_sph_force(enc);
        }
    }

    /// Append the resident SPH stages onto `enc`: gather gas positions off `bodies`, build
    /// the gas grid, root-find (ρ, h) — left resident — then (G5b) the hydro force scattered
    /// additively into `accel`'s gas rows. Requires gas mode + a non-empty gas map.
    fn encode_sph_force(&self, enc: &mut wgpu::CommandEncoder) {
        if let Some(sph) = &self.sph {
            sph.encode(enc, self.gas_idx.len());
        }
    }

    /// Copy the resident gas (ρ, h) back to the host paired with the gas map (the G5a gate
    /// surface). Runs after an [`upload`](Self::upload) (its prime evaluates density) or any
    /// [`step`](Self::step). Requires gas mode.
    pub fn snapshot_gas_density(&mut self) -> GasDensity {
        let n_gas = self.gas_idx.len();
        let sph = self
            .sph
            .as_ref()
            .expect("snapshot_gas_density requires gas mode (new_with_sph)");
        if n_gas == 0 {
            return GasDensity {
                gas_idx: Vec::new(),
                rho: Vec::new(),
                h: Vec::new(),
            };
        }
        let (rho, h) = sph.snapshot(&self.core.device, &self.core.queue, n_gas);
        self.submits += 1;
        GasDensity {
            gas_idx: self.gas_idx.clone(),
            rho,
            h,
        }
    }

    /// Read the resident gas hydro acceleration back to the host, in ascending gas-index
    /// order (the G5b gate surface). This is the PURE hydro force `gas_acc` — the value the
    /// hydro pass leaves resident BEFORE the scatter-add folds it into `accel`'s gas rows —
    /// so the isolated momentum-antisymmetry and hydro-accuracy gates see it uncontaminated
    /// by the (non-antisymmetric) gravity contribution. Requires gas mode.
    pub fn snapshot_gas_accel(&mut self) -> Vec<DVec3> {
        todo!("G5b: read the resident gas_acc (pre-scatter hydro force) back to the host")
    }

    /// Read the full resident acceleration buffer (`accel`, all particles) back to the
    /// host. The G5b scatter gate uses this to check the hydro force landed in the gas rows
    /// and left the star rows untouched (gravity-only vs gas-mode stepper).
    pub fn snapshot_accel(&mut self) -> Vec<DVec3> {
        todo!("G5b: read the resident accel buffer back to the host")
    }

    /// Upload `state` (f64→f32 narrowed) into the resident GPU buffers, (re)allocating as `N`
    /// changes, and **prime** the acceleration (one force evaluation, no readback) so the first
    /// [`step`](Self::step)'s opening half-kick uses `a(x₀)`, not a stale value. Resets the clock.
    pub fn upload(&mut self, state: &State) {
        let n = state.len();
        self.n = n;
        self.time = 0.0;
        self.mass = state.mass.clone();
        // Rebuild the gas map from `state.kind` every upload (a re-upload with a different
        // gas/star split must not carry a stale map). Empty in gravity-only mode.
        self.gas_idx = if self.sph.is_some() {
            (0..n).filter(|&i| state.kind[i] == Species::Gas).collect()
        } else {
            Vec::new()
        };
        if n == 0 {
            return;
        }

        let cap = n.max(2); // never size intermediate buffers to zero
        self.core.ensure_capacity(cap);
        self.ensure_capacity(cap);
        self.core.write_uniforms(n);
        self.write_step_params(0.0);

        // Split each f64 position into a double-single `hi + lo` pair: `hi` (bodies.xyz, the force
        // pipeline's f32 input) is the narrowed coordinate, `lo` (pos_lo.xyz) the f64 residual
        // it dropped. Seeding the residual (not zero) captures ~46 bits of the f64 input and makes
        // snapshot↔upload a lossless round-trip — the M4i faithful gate stays exact. `lo` is
        // normalized by construction (|residual| ≤ ½ulp(hi)).
        let split = |x: f64| {
            let hi = x as f32;
            (hi, (x - hi as f64) as f32)
        };
        let mut bodies = Vec::with_capacity(n);
        let mut pos_los = Vec::with_capacity(n);
        for i in 0..n {
            let p = state.pos[i];
            let (hx, lx) = split(p.x);
            let (hy, ly) = split(p.y);
            let (hz, lz) = split(p.z);
            bodies.push([hx, hy, hz, state.mass[i] as f32]);
            pos_los.push([lx, ly, lz, 0.0]);
        }
        let vels: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let v = state.vel[i];
                [v.x as f32, v.y as f32, v.z as f32, 0.0]
            })
            .collect();
        {
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            let res = self.res.as_ref().expect("resident resources ensured");
            self.core
                .queue
                .write_buffer(&core_res.bodies, 0, bytemuck::cast_slice(&bodies));
            self.core
                .queue
                .write_buffer(&res.pos_lo, 0, bytemuck::cast_slice(&pos_los));
            self.core
                .queue
                .write_buffer(&res.vel, 0, bytemuck::cast_slice(&vels));
        }

        // Gas mode: (re)allocate the SPH buffers, write the density seed params + gas map,
        // and rebuild the SPH bind groups against the just-written `bodies`. Must precede the
        // prime so `a(x₀)` already includes the gas density (→ hydro, G5b).
        if !self.gas_idx.is_empty() {
            let gas_pos: Vec<DVec3> = self.gas_idx.iter().map(|&i| state.pos[i]).collect();
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            let sph = self.sph.as_mut().expect("gas map non-empty ⇒ gas mode");
            sph.prepare(
                &self.core.device,
                &self.core.queue,
                &core_res.bodies,
                &self.gas_idx,
                &gas_pos,
            );
        }

        // Prime accel = a(x₀). No readback.
        let mut enc = self
            .core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident-prime-encoder"),
            });
        self.encode_force(&mut enc);
        self.core.queue.submit([enc.finish()]);
        self.submits += 1;
    }

    /// Append one resident KDK step to `enc` (no submit): kick½ · drift · (reset+build+traverse
    /// into `acc`) · kick½. Requires `write_step_params(dt)` already written and `n >= 1`. Chaining
    /// several of these into one encoder is exactly [`step_many`](Self::step_many)'s batch — because
    /// each pass is its own compute pass, wgpu's usage tracking inserts the read-after-write
    /// barriers *between steps* too (drift writes `bodies` → the next step's force reads it; the
    /// closing kick writes `vel` → the next drift reads it), so batching regroups encoders without
    /// touching the arithmetic. The two half-kicks are **kept separate** across the step boundary:
    /// `kick½(a)·kick½(a)` is *not* f32-identical to a fused `kick(a·dt)`, so fusing them would
    /// silently change the trajectory and break the faithful gate.
    fn encode_one_step(&self, enc: &mut wgpu::CommandEncoder) {
        let res = self.res.as_ref().expect("resident resources ensured");
        // Kick½ with a(xₙ) [carried from the previous step's closing kick, or the prime].
        self.per_particle_pass(enc, "resident-kick-open", &self.kick_pl, &res.kick_bg);
        // Drift: xₙ → xₙ₊₁.
        self.per_particle_pass(enc, "resident-drift", &self.drift_pl, &res.drift_bg);
        // Recompute a(xₙ₊₁) in place (accel), left resident for the next step's opening kick.
        self.encode_force(enc);
        // Kick½ with a(xₙ₊₁).
        self.per_particle_pass(enc, "resident-kick-close", &self.kick_pl, &res.kick_bg);
    }

    /// Advance one resident KDK step by `dt` in one submit, no readback. Requires a prior
    /// [`upload`](Self::upload). This is the minimum-latency path; [`step_many`](Self::step_many)
    /// batches several steps per submit for throughput.
    pub fn step(&mut self, dt: f64) {
        if self.n == 0 {
            self.time += dt;
            return;
        }
        self.write_step_params(dt);
        let mut enc = self
            .core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident-step-encoder"),
            });
        self.encode_one_step(&mut enc);
        self.core.queue.submit([enc.finish()]);
        self.submits += 1;

        self.time += dt;
    }

    /// Advance `steps` resident KDK steps of `dt`, batching up to [`MAX_BATCH`](Self::MAX_BATCH)
    /// steps into a **single encoder/submit** — `⌈steps/MAX_BATCH⌉` submits total, dropping the
    /// per-step submit overhead (the named M4i throughput follow-up) while keeping each submit
    /// bounded under the OS GPU watchdog. `dt` is constant across the run, so the per-step uniform
    /// is written **once**; the trajectory is identical to `steps` individual [`step`](Self::step)s
    /// (only the submit grouping changes — see [`encode_one_step`](Self::encode_one_step)).
    pub fn step_many(&mut self, dt: f64, steps: u64) {
        if self.n == 0 {
            self.time += steps as f64 * dt;
            return;
        }
        if steps == 0 {
            return;
        }
        self.write_step_params(dt);
        let mut remaining = steps;
        while remaining > 0 {
            let chunk = remaining.min(Self::MAX_BATCH);
            let mut enc =
                self.core
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("resident-batch-encoder"),
                    });
            for _ in 0..chunk {
                self.encode_one_step(&mut enc);
            }
            self.core.queue.submit([enc.finish()]);
            self.submits += 1;
            self.time += chunk as f64 * dt;
            remaining -= chunk;
        }
    }

    /// Read the resident state back to the host as a fresh [`State`] (pos/vel widened f32→f64,
    /// mass/time host-tracked). The only device→host transfer.
    pub fn snapshot(&mut self) -> State {
        let n = self.n;
        if n == 0 {
            let mut s = State::from_phase_space(vec![], vec![], vec![]);
            s.time = self.time;
            return s;
        }

        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        let (pos, vel) = {
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            let res = self.res.as_ref().expect("resident resources ensured");

            let mut enc =
                self.core
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("resident-snapshot-encoder"),
                    });
            enc.copy_buffer_to_buffer(&core_res.bodies, 0, &res.pos_readback, 0, bytes);
            enc.copy_buffer_to_buffer(&res.pos_lo, 0, &res.pos_lo_readback, 0, bytes);
            enc.copy_buffer_to_buffer(&res.vel, 0, &res.vel_readback, 0, bytes);
            self.core.queue.submit([enc.finish()]);
            self.submits += 1;

            // Recombine the double-single: pos = (f64)hi + (f64)lo. Both are f32, so the sum is
            // exact in f64 — this is where the accumulated sub-f32 precision reaches the host.
            let hi = self.read_vec3(&res.pos_readback, bytes);
            let lo = self.read_vec3(&res.pos_lo_readback, bytes);
            let pos = hi.iter().zip(&lo).map(|(&h, &l)| h + l).collect();
            let vel = self.read_vec3(&res.vel_readback, bytes);
            (pos, vel)
        };

        let mut s = State::from_phase_space(pos, vel, self.mass.clone());
        s.time = self.time;
        s
    }

    /// Map a f32 `vec4` buffer, block once, and widen the xyz lanes of the first `n` entries to
    /// f64. A map failure is an exceptional GPU loss and panics rather than return corrupt state.
    fn read_vec3(&self, readback: &wgpu::Buffer, bytes: u64) -> Vec<DVec3> {
        let slice = readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        rx.recv()
            .expect("map channel closed")
            .expect("gpu buffer map failed");

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let out = (0..self.n)
            .map(|i| {
                let b = i * 4;
                DVec3::new(floats[b] as f64, floats[b + 1] as f64, floats[b + 2] as f64)
            })
            .collect();
        drop(data);
        readback.unmap();
        out
    }

    /// Simulation time after the steps taken so far.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Total `queue.submit`s issued over this stepper's life (the prime in [`upload`](Self::upload),
    /// every [`step`](Self::step)/[`step_many`](Self::step_many) flush, and each
    /// [`snapshot`](Self::snapshot)). Exposed so the batching gate can assert `step_many` coalesces
    /// its steps into `⌈steps/MAX_BATCH⌉` submits rather than one-per-step.
    pub fn submits(&self) -> u64 {
        self.submits
    }

    /// Number of resident particles (0 before the first [`upload`](Self::upload)).
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether no particles are resident.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}
