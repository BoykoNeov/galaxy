//! [`GpuDirectSum`]: exact O(N²) Plummer-softened direct summation on the GPU.
//!
//! Holds a reusable wgpu compute context (adapter/device/queue + pipeline) built
//! once and driven per `accelerations` call; storage buffers grow lazily with N.
//! See the crate docs for the precision (f32-forced) and determinism (gather, not
//! scatter) rationale.

use bytemuck::{Pod, Zeroable};

use galaxy_core::{DVec3, ForceSolver, State};

use crate::GpuError;

/// Compute workgroup size = the shared-memory tile width. 256 is within the
/// baseline wgpu limit (`maxComputeInvocationsPerWorkgroup`), and 256·16 B = 4 KiB
/// of workgroup storage is far under the 16 KiB floor.
const WORKGROUP_SIZE: u32 = 256;

/// Tiled **gather** direct-sum kernel. One invocation per *target* `i` accumulates
/// the force from every source `j` into a private register and writes `accel[i]`
/// exactly once — no float `atomicAdd`, so the loop order is fixed and the result is
/// bit-deterministic on a given device. Sources are staged through a workgroup tile
/// (the classic GPU-Gems N-body pattern) to amortize global-memory traffic.
///
/// The self term (`j == i`) and padded lanes (`j >= n`, mass 0) both contribute zero
/// — `dx = 0` and `mass = 0` respectively — so no per-iteration branch is needed.
const SHADER: &str = r#"
struct Params {
    n: u32,
    eps2: f32,
    g: f32,
    pad: f32,
};

@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;      // xyz = pos, w = mass
@group(0) @binding(1) var<storage, read_write> accel: array<vec4<f32>>; // xyz = accel
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> tile: array<vec4<f32>, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;

    var pi = vec3<f32>(0.0, 0.0, 0.0);
    if (i < n) { pi = bodies[i].xyz; }

    var a = vec3<f32>(0.0, 0.0, 0.0);
    let ntiles = (n + WG - 1u) / WG;
    for (var t: u32 = 0u; t < ntiles; t = t + 1u) {
        let j = t * WG + lid.x;
        if (j < n) { tile[lid.x] = bodies[j]; }
        else { tile[lid.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0); }
        workgroupBarrier();

        for (var k: u32 = 0u; k < WG; k = k + 1u) {
            let bj = tile[k];
            let dx = bj.xyz - pi;
            let r2 = dot(dx, dx) + params.eps2;
            let inv_r = inverseSqrt(r2);
            let inv_r3 = inv_r * inv_r * inv_r;   // (r² + ε²)^(-3/2)
            a = a + (bj.w * inv_r3) * dx;         // mⱼ · dx / r³
        }
        workgroupBarrier();
    }

    if (i < n) { accel[i] = vec4<f32>(a * params.g, 0.0); }
}
"#;

/// Uniform block mirroring the WGSL `Params` (16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    eps2: f32,
    g: f32,
    _pad: f32,
}

/// GPU direct-summation force solver. Same `(g, softening)` semantics as
/// [`galaxy_solvers::DirectSum`], evaluated in an f32 wgpu compute kernel.
pub struct GpuDirectSum {
    /// Gravitational constant.
    g: f64,
    /// Plummer softening length ε.
    softening: f64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    // Storage grown lazily to the largest N seen; the bind group is rebuilt with it.
    bodies_buf: Option<wgpu::Buffer>,
    accel_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    capacity: usize,
}

impl GpuDirectSum {
    /// Bring up a headless wgpu compute device and build the direct-sum pipeline.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available.
    ///
    /// Requests **no** device features — compute over storage buffers is baseline, so
    /// this does not narrow adapter support (unlike the renderer's `FLOAT32_BLENDABLE`).
    pub fn new(g: f64, softening: f64) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(g, softening))
    }

    async fn new_async(g: f64, softening: f64) -> Result<Self, GpuError> {
        let crate::context::GpuContext { device, queue } = crate::context::gpu_context()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-direct-sum-shader"),
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
            label: Some("gpu-direct-sum-bgl"),
            entries: &[
                storage(0, true),  // bodies (read)
                storage(1, false), // accel (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("gpu-direct-sum-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-direct-sum-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-direct-sum-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuDirectSum {
            g,
            softening,
            device,
            queue,
            pipeline,
            bgl,
            params_buf,
            bodies_buf: None,
            accel_buf: None,
            readback_buf: None,
            bind_group: None,
            capacity: 0,
        })
    }

    /// Ensure the storage/readback buffers hold at least `n` bodies, (re)building the
    /// bind group when they are (re)allocated. Only grows — a later smaller N reuses
    /// the larger buffers (the kernel bounds itself by `params.n`, and only `n·16` B
    /// is ever copied back). Caller guarantees `n > 0`.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity && self.bind_group.is_some() {
            return;
        }
        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        let bodies = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-direct-sum-bodies"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let accel = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-direct-sum-accel"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-direct-sum-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-direct-sum-bind-group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bodies.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });
        self.bodies_buf = Some(bodies);
        self.accel_buf = Some(accel);
        self.readback_buf = Some(readback);
        self.bind_group = Some(bind_group);
        self.capacity = n;
    }
}

impl ForceSolver for GpuDirectSum {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        self.ensure_capacity(n);

        // Pack positions (f32) + mass into vec4 bodies — this f64→f32 narrowing is the
        // precision reduction the crate docs own.
        let bodies: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let p = state.pos[i];
                [p.x as f32, p.y as f32, p.z as f32, state.mass[i] as f32]
            })
            .collect();
        let params = Params {
            n: n as u32,
            eps2: (self.softening * self.softening) as f32,
            g: self.g as f32,
            _pad: 0.0,
        };

        let bodies_buf = self.bodies_buf.as_ref().expect("buffers ensured above");
        let accel_buf = self.accel_buf.as_ref().expect("buffers ensured above");
        let readback = self.readback_buf.as_ref().expect("buffers ensured above");
        let bind_group = self.bind_group.as_ref().expect("bind group ensured above");

        self.queue
            .write_buffer(bodies_buf, 0, bytemuck::cast_slice(&bodies));
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-direct-sum-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(accel_buf, 0, readback, 0, bytes);
        self.queue.submit([enc.finish()]);

        // Map, block until the GPU is done, widen f32 accelerations back to f64.
        // `new()` already validated the device; a map failure here is a genuinely
        // exceptional GPU loss, so — like the CPU solvers' `assert` on this infallible
        // trait path — it panics rather than silently corrupting the state.
        let slice = readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        rx.recv()
            .expect("map channel closed")
            .expect("gpu buffer map failed");

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        for (i, a) in acc.iter_mut().enumerate() {
            let b = i * 4;
            *a = DVec3::new(floats[b] as f64, floats[b + 1] as f64, floats[b + 2] as f64);
        }
        drop(data);
        readback.unmap();
    }

    /// Softened potential energy, delegated to the shared CPU **f64** reduction.
    ///
    /// NOTE (documented inconsistency): `accelerations` applies **f32** forces while
    /// this reports an **f64** potential, so an energy-drift diagnostic over a GPU run
    /// mixes a force/potential *precision* gap with integrator error. Acceptable for
    /// the MVP — the potential is a periodic O(N²) diagnostic, not the per-step path.
    fn potential_energy(&self, state: &State) -> f64 {
        galaxy_solvers::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}
