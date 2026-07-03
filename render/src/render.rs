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
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 1920,
            height: 1080,
            falloff: 6.0,
            min_splat_px: 1.5,
            max_splat_ndc: 1.0,
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

const SHADER: &str = r#"
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
fn vs(@location(0) corner: vec2<f32>,
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
    out.emissive = emissive * dim;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let g = exp(-u.params.y * dot(in.local, in.local));
    return vec4<f32>(in.emissive * g, g);
}
"#;

/// Unit quad (two triangles) in local [-1, 1] space, scaled per splat by `half`.
const QUAD: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

/// The reusable GPU rendering context: adapter/device/queue + the splat pipeline,
/// created once and driven for every frame of a movie.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    quad_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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

        Ok(Renderer {
            device,
            queue,
            pipeline,
            quad_buf,
            uniform_buf,
            bind_group,
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
        todo!("M7e: prepass → attenuated stars → gas march, one additive target")
    }

    /// Render one frame: additively blend every particle in `frame` as a Gaussian
    /// splat, as seen by `camera`, into an `Rgba32Float` target of
    /// `cfg.width × cfg.height`, and read it back as a linear [`HdrImage`].
    pub fn render_frame(
        &self,
        frame: &FrameData,
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
                    contents: bytemuck::cast_slice(&splats),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });

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
            if let Some(inst) = &instance_buf {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_buf.slice(..));
                pass.set_vertex_buffer(1, inst.slice(..));
                pass.draw(0..6, 0..splats.len() as u32);
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
}

/// Additive blend factor pair: `dst = src·1 + dst·1` (order-independent accumulation).
const ADDITIVE: wgpu::BlendComponent = wgpu::BlendComponent {
    src_factor: wgpu::BlendFactor::One,
    dst_factor: wgpu::BlendFactor::One,
    operation: wgpu::BlendOperation::Add,
};
