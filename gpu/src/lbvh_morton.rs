//! [`GpuMortonBuilder`]: the GPU Morton + bounding-box build stage — stage 1 of the
//! GPU-resident LBVH build (DESIGN M4c).
//!
//! Given particle positions it computes, on the GPU (wgpu compute, **f32**), the root
//! bounding box and the per-particle 30-bit Morton codes + their three quantized lanes.
//! It is the GPU port of the prologue of [`galaxy_solvers::LbvhFlat::build`] and is gated
//! directly against the CPU reference [`galaxy_solvers::reference_morton`].
//!
//! ## Two passes, f32 throughout
//! 1. **bbox reduction** — a single workgroup grid-strides all positions into per-lane
//!    private min/max, then a fixed-order shared-memory tree reduction folds to lane 0,
//!    which writes the bbox. min/max never round and are order-independent, so this is
//!    bit-exact and deterministic with no float atomics (which WGSL lacks anyway).
//! 2. **quantize** — one invocation per particle reconstructs the exact CPU bbox
//!    convention (pad/floor/scale) in f32, then floors + clamps each axis to `[0, 1023]`
//!    and interleaves the three lanes into a 30-bit code.
//!
//! ## Scope (state plainly)
//! This proves **quantization + the reduction pattern**. It does **not** prove the tree
//! matches the reference: f32 codes diverge from the f64 reference near cell boundaries
//! (a straddling particle floors into an adjacent 1024³ cell), so the eventual GPU tree
//! *topology* can differ — the expected analogue of the θ-straddle in [`crate::GpuTree`],
//! not a bug. The real correctness check is the later θ→0 physics gate on the deferred
//! `GpuLbvh`. This stage's gates are structural + tolerance + determinism only.

use bytemuck::{Pod, Zeroable};

use galaxy_core::DVec3;

use crate::GpuError;

/// Reduction / quantization workgroup width. 256 is within the baseline wgpu limit and
/// the two `vec3` workgroup arrays (256·16 B each) sit well under the 16 KiB floor.
const WORKGROUP_SIZE: u32 = 256;

/// Two compute entry points sharing one bind group. `reduce` folds the bounding box in a
/// single workgroup (grid-stride → shared-memory tree reduce, fixed order → deterministic
/// and bit-exact, no float atomics). `quantize` reconstructs the **exact** CPU bbox
/// convention in f32 (`center`, `half = max(0.5·ext, 1e-12)·(1+1e-9)`, `scale = 1024/size`)
/// and emits each particle's three clamped lanes + interleaved 30-bit code.
///
/// The `(1+1e-9)` pad folds to `1.0` in f32 (below the ulp at 1.0); harmless — the top-edge
/// particle is caught by the `min(1023)` clamp instead of the pad's nudge, a ≤1-lane effect
/// the reference-agreement gate tolerates.
///
/// `pub(crate)` so the M4h [`crate::GpuLbvhFused`] can compile the *same* `reduce`/`quantize`
/// kernels onto its single fused device (one source of truth → the fuse runs identical code).
pub(crate) const SHADER: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
struct BBox { lo: vec4<f32>, hi: vec4<f32> };

@group(0) @binding(0) var<storage, read>       positions: array<vec4<f32>>; // xyz = pos
@group(0) @binding(1) var<storage, read_write> bbox: BBox;
@group(0) @binding(2) var<storage, read_write> lanes: array<vec4<u32>>;      // xyz = lanes
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<uniform>             params: Params;

const WG: u32 = 256u;
const FMAX: f32 = 3.4028235e38;

var<workgroup> smin: array<vec3<f32>, WG>;
var<workgroup> smax: array<vec3<f32>, WG>;

// --- pass 1: single-workgroup bounding-box reduction -----------------------
@compute @workgroup_size(WG)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = params.n;
    var lo = vec3<f32>( FMAX,  FMAX,  FMAX);
    var hi = vec3<f32>(-FMAX, -FMAX, -FMAX);
    // Grid-stride over the whole array in fixed order (min/max are order-independent).
    var i = lid.x;
    loop {
        if (i >= n) { break; }
        let p = positions[i].xyz;
        lo = min(lo, p);
        hi = max(hi, p);
        i = i + WG;
    }
    smin[lid.x] = lo;
    smax[lid.x] = hi;
    workgroupBarrier();
    // Fixed-order tree reduction to lane 0.
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            smin[lid.x] = min(smin[lid.x], smin[lid.x + stride]);
            smax[lid.x] = max(smax[lid.x], smax[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        bbox.lo = vec4<f32>(smin[0], 0.0);
        bbox.hi = vec4<f32>(smax[0], 0.0);
    }
}

// --- Morton primitives (mirror galaxy_solvers::lbvh) -----------------------
fn expand10(vv: u32) -> u32 {
    var x = vv & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u))  & 0x0300f00fu;
    x = (x | (x << 4u))  & 0x030c30c3u;
    x = (x | (x << 2u))  & 0x09249249u;
    return x;
}
fn morton3(x: u32, y: u32, z: u32) -> u32 {
    return expand10(x) | (expand10(y) << 1u) | (expand10(z) << 2u);
}
// floor → clamp to [0, 1023], matching (v.floor().max(0.0) as u32).min(1023).
fn quant_lane(v: f32) -> u32 {
    let f = floor(v);
    if (f < 0.0) { return 0u; }
    if (f > 1023.0) { return 1023u; }
    return u32(f);
}

// --- pass 2: reconstruct bounds + quantize ---------------------------------
@compute @workgroup_size(WG)
fn quantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let lo = bbox.lo.xyz;
    let hi = bbox.hi.xyz;
    let center = 0.5 * (lo + hi);
    let d = hi - lo;
    let ext = max(max(d.x, d.y), d.z);
    let half = max(0.5 * ext, 1e-12) * (1.0 + 1e-9);
    let bmin = center - vec3<f32>(half, half, half);
    let scale = 1024.0 / (2.0 * half);
    let rel = (positions[i].xyz - bmin) * scale;
    let lx = quant_lane(rel.x);
    let ly = quant_lane(rel.y);
    let lz = quant_lane(rel.z);
    lanes[i] = vec4<u32>(lx, ly, lz, 0u);
    codes[i] = morton3(lx, ly, lz);
}
"#;

/// Uniform block mirroring the WGSL `Params` (16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad: [u32; 3],
}

/// Result of the GPU Morton+bbox stage. `bbox_min`/`bbox_max` are the raw f32 reduction
/// output (for the reduction gate); `lanes`/`codes` are the quantized output feeding the
/// next stage (the GPU sort).
pub struct GpuMorton {
    /// Raw bounding-box low corner from the GPU reduction (f32, per axis).
    pub bbox_min: [f32; 3],
    /// Raw bounding-box high corner from the GPU reduction (f32, per axis).
    pub bbox_max: [f32; 3],
    /// Per-particle quantized lanes `[x, y, z]`, each in `[0, 1024)`.
    pub lanes: Vec<[u32; 3]>,
    /// Per-particle interleaved 30-bit Morton codes.
    pub codes: Vec<u32>,
}

/// GPU Morton + bounding-box build stage. Holds a reusable wgpu compute context built
/// once and driven per [`compute`](Self::compute) call; storage buffers grow lazily with
/// N. Same bring-up idiom as [`crate::GpuDirectSum`] (baseline storage-buffer compute, no
/// device features → no adapter narrowing).
pub struct GpuMortonBuilder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    reduce_pipeline: wgpu::ComputePipeline,
    quantize_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    bbox_buf: wgpu::Buffer,
    bbox_readback: wgpu::Buffer,
    // Storage grown lazily to the largest N seen; the bind group is rebuilt with it.
    pos_buf: Option<wgpu::Buffer>,
    lanes_buf: Option<wgpu::Buffer>,
    codes_buf: Option<wgpu::Buffer>,
    lanes_readback: Option<wgpu::Buffer>,
    codes_readback: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    capacity: usize,
}

/// Bytes of a `BBox` (two `vec4<f32>`).
const BBOX_BYTES: u64 = 32;

impl GpuMortonBuilder {
    /// Bring up a headless wgpu compute device and build the Morton+bbox pipelines.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // headless
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("galaxy-gpu-morton-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-morton-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-morton-bgl"),
            entries: &[
                storage(0, true),  // positions (read)
                storage(1, false), // bbox (read-write)
                storage(2, false), // lanes (read-write)
                storage(3, false), // codes (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-morton-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let reduce_pipeline = make_pipeline("reduce", "gpu-morton-reduce");
        let quantize_pipeline = make_pipeline("quantize", "gpu-morton-quantize");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bbox_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-bbox"),
            size: BBOX_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bbox_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-bbox-readback"),
            size: BBOX_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(GpuMortonBuilder {
            device,
            queue,
            reduce_pipeline,
            quantize_pipeline,
            bgl,
            params_buf,
            bbox_buf,
            bbox_readback,
            pos_buf: None,
            lanes_buf: None,
            codes_buf: None,
            lanes_readback: None,
            codes_readback: None,
            bind_group: None,
            capacity: 0,
        })
    }

    /// Ensure the per-particle storage/readback buffers hold at least `n` bodies,
    /// (re)building the bind group when they are (re)allocated. Only grows. Caller
    /// guarantees `n > 0`.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity && self.bind_group.is_some() {
            return;
        }
        let pos_bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        let lane_bytes = (n * std::mem::size_of::<[u32; 4]>()) as u64;
        let code_bytes = (n * std::mem::size_of::<u32>()) as u64;

        let pos = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-positions"),
            size: pos_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lanes = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-lanes"),
            size: lane_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let codes = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-codes"),
            size: code_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lanes_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-lanes-readback"),
            size: lane_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let codes_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-morton-codes-readback"),
            size: code_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-morton-bind-group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bbox_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lanes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: codes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });

        self.pos_buf = Some(pos);
        self.lanes_buf = Some(lanes);
        self.codes_buf = Some(codes);
        self.lanes_readback = Some(lanes_readback);
        self.codes_readback = Some(codes_readback);
        self.bind_group = Some(bind_group);
        self.capacity = n;
    }

    /// Compute the bounding box + Morton codes for `pos` on the GPU. `pos` may be empty
    /// (yields empty `lanes`/`codes` and a degenerate bbox with no dispatch).
    pub fn compute(&mut self, pos: &[DVec3]) -> GpuMorton {
        let n = pos.len();
        if n == 0 {
            return GpuMorton {
                bbox_min: [0.0; 3],
                bbox_max: [0.0; 3],
                lanes: Vec::new(),
                codes: Vec::new(),
            };
        }
        self.ensure_capacity(n);

        // Narrow positions f64 → f32 (the toolchain-forced precision reduction).
        let bodies: Vec<[f32; 4]> = pos
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
            .collect();
        let params = Params {
            n: n as u32,
            _pad: [0; 3],
        };

        let pos_buf = self.pos_buf.as_ref().expect("buffers ensured above");
        let lanes_buf = self.lanes_buf.as_ref().expect("buffers ensured above");
        let codes_buf = self.codes_buf.as_ref().expect("buffers ensured above");
        let lanes_rb = self.lanes_readback.as_ref().expect("buffers ensured above");
        let codes_rb = self.codes_readback.as_ref().expect("buffers ensured above");
        let bind_group = self.bind_group.as_ref().expect("bind group ensured above");

        self.queue
            .write_buffer(pos_buf, 0, bytemuck::cast_slice(&bodies));
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let lane_bytes = (n * std::mem::size_of::<[u32; 4]>()) as u64;
        let code_bytes = (n * std::mem::size_of::<u32>()) as u64;

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-morton-reduce-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single-workgroup reduction
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-morton-quantize-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.quantize_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.bbox_buf, 0, &self.bbox_readback, 0, BBOX_BYTES);
        enc.copy_buffer_to_buffer(lanes_buf, 0, lanes_rb, 0, lane_bytes);
        enc.copy_buffer_to_buffer(codes_buf, 0, codes_rb, 0, code_bytes);
        self.queue.submit([enc.finish()]);

        // Map all three readbacks, wait once, read. A map failure here is a genuine GPU
        // loss (new() already validated the device), so — like the CPU solvers' asserts
        // on infallible paths — it panics rather than silently corrupting state.
        let bbox_slice = self.bbox_readback.slice(..BBOX_BYTES);
        let lanes_slice = lanes_rb.slice(..lane_bytes);
        let codes_slice = codes_rb.slice(..code_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        for slice in [&bbox_slice, &lanes_slice, &codes_slice] {
            let tx = tx.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        for _ in 0..3 {
            rx.recv()
                .expect("map channel closed")
                .expect("gpu buffer map failed");
        }

        let bbox_data = bbox_slice.get_mapped_range();
        let bbox_f: &[f32] = bytemuck::cast_slice(&bbox_data);
        let bbox_min = [bbox_f[0], bbox_f[1], bbox_f[2]];
        let bbox_max = [bbox_f[4], bbox_f[5], bbox_f[6]];

        let lanes_data = lanes_slice.get_mapped_range();
        let lanes_u: &[u32] = bytemuck::cast_slice(&lanes_data);
        let lanes: Vec<[u32; 3]> = (0..n)
            .map(|i| [lanes_u[i * 4], lanes_u[i * 4 + 1], lanes_u[i * 4 + 2]])
            .collect();

        let codes_data = codes_slice.get_mapped_range();
        let codes: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&codes_data)[..n].to_vec();

        drop(bbox_data);
        drop(lanes_data);
        drop(codes_data);
        self.bbox_readback.unmap();
        lanes_rb.unmap();
        codes_rb.unmap();

        GpuMorton {
            bbox_min,
            bbox_max,
            lanes,
            codes,
        }
    }
}
