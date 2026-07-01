//! [`GpuTree`]: Barnes-Hut O(N log N) tree force solver — CPU build, GPU traverse.
//!
//! The octree is built and linearized on the CPU (reusing the tested
//! [`galaxy_solvers::FlatTree`] — DFS pre-order with skip pointers), uploaded, and
//! walked on the GPU by a **stackless** gather kernel: one invocation per target
//! follows the skip pointers with a single node index and no recursion stack.
//!
//! ## Precision: f32, same story as [`crate::GpuDirectSum`]
//! wgpu/naga has no portable f64 compute, so the kernel is f32 while the engine is
//! f64. The tree geometry (center/half/com/delta) narrows f64→f32 (small, O(1e-6)
//! relative); the dominant error is the same as the direct-sum kernel — catastrophic
//! cancellation in `xᵢ − xⱼ` and terms swallowed in the f32 accumulator, worst in the
//! large-coordinate collision regime. Keep collision coordinates near the origin.
//!
//! ## Determinism: gather + fixed skip-pointer order
//! One invocation per target writes `acc[i]` once from a private accumulator, walking
//! the flat tree in a fixed order — so it is bit-deterministic **on a given device**
//! (no float `atomicAdd`). Cross-device equality is not claimed (FMA/rounding differ).
//!
//! ## Reassociation vs the CPU BarnesHut
//! The stackless walk keeps one running accumulator over the DFS scan, whereas the CPU
//! [`galaxy_solvers::BarnesHut`] folds each subtree separately then combines — a
//! different (valid) summation order. Even in exact arithmetic the two agree only to
//! reassociation precision, not bit-for-bit (the f64 analogue is pinned by a
//! solvers-side tolerance test; cf. `potential_energy_parallel`). In f32 the opening
//! *decision* also differs near threshold, flipping a few nodes — a discrete O(θ²)
//! swing for those targets. Both are documented in the gpu-tree gates.
//!
//! ## Scope
//! A genuine GPU tree *traversal* (the part that dominates at scale), but the build
//! stays on the CPU (already rayon-parallel) and the state is re-uploaded each call —
//! a GPU-resident build (Morton/LBVH) is the next deferred step. Realistically opens
//! the 10⁷ band that O(N²) [`crate::GpuDirectSum`] cannot; the CPU build becomes the
//! Amdahl ceiling well before 10⁸.

use bytemuck::{Pod, Zeroable};

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::FlatTree;

use crate::GpuError;

/// Compute workgroup size. Tree traversal is irregular (no shared-memory tile like
/// the direct-sum kernel), so 64 — a comfortable occupancy width within baseline
/// limits — is used rather than the 256 tile of `GpuDirectSum`.
const WORKGROUP_SIZE: u32 = 64;

/// Stackless **gather** Barnes-Hut kernel. One invocation per target `i` walks the
/// flattened octree with a single `node` index: `mass ≤ 0` skips via the node's skip
/// pointer; a leaf (`body_count > 0`) direct-sums its bodies (excluding self); an
/// internal node is accepted as a monopole when the Barnes (1994) criterion holds
/// (jump to the skip pointer) or opened to its first child (`node + 1`). Because a
/// correct flatten makes `node` strictly increase every step, the walk terminates in
/// ≤ `n_nodes` steps with no stack and no possibility of an infinite loop.
const SHADER: &str = r#"
struct Params {
    n: u32,
    n_nodes: u32,
    eps2: f32,
    g: f32,
    theta: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;      // xyz = pos, w = mass
@group(0) @binding(1) var<storage, read> node_ch: array<vec4<f32>>;     // xyz = center, w = half
@group(0) @binding(2) var<storage, read> node_cm: array<vec4<f32>>;     // xyz = com,    w = mass
@group(0) @binding(3) var<storage, read> node_meta: array<vec4<u32>>;   // x=delta bits, y=next, z=body_start, w=body_count
@group(0) @binding(4) var<storage, read> leaf_bodies: array<u32>;       // leaf body original indices
@group(0) @binding(5) var<storage, read_write> accel: array<vec4<f32>>; // xyz = accel
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let xp = bodies[i].xyz;
    let eps2 = params.eps2;
    let theta = params.theta;

    var a = vec3<f32>(0.0, 0.0, 0.0);
    var node: u32 = 0u;
    loop {
        if (node >= params.n_nodes) { break; }
        let cm = node_cm[node];
        let md = node_meta[node];
        let mass = cm.w;
        let next = md.y;
        if (mass <= 0.0) { node = next; continue; }

        let bcount = md.w;
        if (bcount > 0u) {
            // Leaf: exact direct sum over its bodies, excluding the self term.
            let bstart = md.z;
            for (var k: u32 = 0u; k < bcount; k = k + 1u) {
                let b = leaf_bodies[bstart + k];
                if (b != i) {
                    let bj = bodies[b];
                    let dx = bj.xyz - xp;
                    let r2 = dot(dx, dx) + eps2;
                    let inv = inverseSqrt(r2);
                    a = a + (bj.w * inv * inv * inv) * dx;
                }
            }
            node = next;
        } else {
            // Internal: never approximate a cell that contains the target.
            let ch = node_ch[node];
            let center = ch.xyz;
            let half = ch.w;
            let inside = all(abs(xp - center) <= vec3<f32>(half, half, half));
            let com = cm.xyz;
            let dx = com - xp;
            let d2 = dot(dx, dx);
            let d = sqrt(d2);
            let s = 2.0 * half;
            let delta = bitcast<f32>(md.x);
            // Barnes (1994): accept the monopole when s ≤ θ·(d − delta).
            if (!inside && theta * (d - delta) >= s) {
                let r2 = d2 + eps2;
                let inv = inverseSqrt(r2);
                a = a + (mass * inv * inv * inv) * dx;
                node = next;               // skip the subtree
            } else {
                node = node + 1u;          // open: descend to the first child
            }
        }
    }

    accel[i] = vec4<f32>(a * params.g, 0.0);
}
"#;

/// Uniform block mirroring the WGSL `Params` (32 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    n_nodes: u32,
    eps2: f32,
    g: f32,
    theta: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU Barnes-Hut tree force solver. Same `(g, softening, theta)` semantics as
/// [`galaxy_solvers::BarnesHut`], evaluated by an f32 stackless wgpu compute kernel.
pub struct GpuTree {
    /// Gravitational constant.
    g: f64,
    /// Plummer softening length ε.
    softening: f64,
    /// Opening angle θ.
    theta: f64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    // N-sized buffers (bodies, leaf indices, accel, readback) grow with particle
    // count; node buffers grow with the tree's node count. Both grow lazily; the
    // bind group is rebuilt whenever any buffer is reallocated.
    bodies_buf: Option<wgpu::Buffer>,
    leaf_buf: Option<wgpu::Buffer>,
    accel_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,
    node_ch_buf: Option<wgpu::Buffer>,
    node_cm_buf: Option<wgpu::Buffer>,
    node_meta_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    body_capacity: usize,
    node_capacity: usize,
}

impl GpuTree {
    /// Bring up a headless wgpu compute device and build the tree-traversal pipeline.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available.
    ///
    /// Requests **no** device features (baseline storage-buffer compute) and uses
    /// 6 storage + 1 uniform binding, within the default `maxStorageBuffersPerShaderStage`
    /// of 8 — so, like `GpuDirectSum`, it does not narrow adapter support.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(g, softening, theta))
    }

    async fn new_async(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
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
                label: Some("galaxy-gpu-tree-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-tree-shader"),
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
            label: Some("gpu-tree-bgl"),
            entries: &[
                storage(0, true),  // bodies
                storage(1, true),  // node center/half
                storage(2, true),  // node com/mass
                storage(3, true),  // node meta
                storage(4, true),  // leaf bodies
                storage(5, false), // accel (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            label: Some("gpu-tree-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-tree-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-tree-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuTree {
            g,
            softening,
            theta,
            device,
            queue,
            pipeline,
            bgl,
            params_buf,
            bodies_buf: None,
            leaf_buf: None,
            accel_buf: None,
            readback_buf: None,
            node_ch_buf: None,
            node_cm_buf: None,
            node_meta_buf: None,
            bind_group: None,
            body_capacity: 0,
            node_capacity: 0,
        })
    }

    /// Ensure the N-sized and node-sized buffers hold at least `n` bodies and
    /// `n_nodes` nodes, (re)building the bind group when any buffer is reallocated.
    /// Only grows — a later smaller problem reuses the larger buffers (the kernel
    /// bounds itself by `params.n` / `params.n_nodes`). Caller guarantees `n > 0` and
    /// `n_nodes > 0`.
    fn ensure_capacity(&mut self, n: usize, n_nodes: usize) {
        let mut rebuild = self.bind_group.is_none();

        if n > self.body_capacity {
            let vec4_bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
            self.bodies_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-bodies"),
                size: vec4_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.leaf_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-leaf-bodies"),
                size: (n * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.accel_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-accel"),
                size: vec4_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.readback_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-readback"),
                size: vec4_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.body_capacity = n;
            rebuild = true;
        }

        if n_nodes > self.node_capacity {
            let f4 = (n_nodes * std::mem::size_of::<[f32; 4]>()) as u64;
            let u4 = (n_nodes * std::mem::size_of::<[u32; 4]>()) as u64;
            self.node_ch_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-node-ch"),
                size: f4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_cm_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-node-cm"),
                size: f4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_meta_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tree-node-meta"),
                size: u4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_capacity = n_nodes;
            rebuild = true;
        }

        if rebuild {
            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu-tree-bind-group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.bodies_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.node_ch_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.node_cm_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.node_meta_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.leaf_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.accel_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.params_buf.as_entire_binding(),
                    },
                ],
            }));
        }
    }
}

impl ForceSolver for GpuTree {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }

        // CPU build + linearize (bit-identical ParallelExact tree, DFS + skip
        // pointers). Every leaf holds ≥1 body, so `leaf_bodies.len() == n`.
        let flat = FlatTree::build(&state.pos, &state.mass);
        let n_nodes = flat.nodes.len();
        self.ensure_capacity(n, n_nodes);

        // Pack GPU arrays. The f64→f32 narrowing here is the precision reduction the
        // crate docs own; delta rides as raw f32 bits in the meta block.
        let bodies: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let p = state.pos[i];
                [p.x as f32, p.y as f32, p.z as f32, state.mass[i] as f32]
            })
            .collect();
        let node_ch: Vec<[f32; 4]> = flat
            .nodes
            .iter()
            .map(|nd| {
                [
                    nd.center.x as f32,
                    nd.center.y as f32,
                    nd.center.z as f32,
                    nd.half as f32,
                ]
            })
            .collect();
        let node_cm: Vec<[f32; 4]> = flat
            .nodes
            .iter()
            .map(|nd| {
                [
                    nd.com.x as f32,
                    nd.com.y as f32,
                    nd.com.z as f32,
                    nd.mass as f32,
                ]
            })
            .collect();
        let node_meta: Vec<[u32; 4]> = flat
            .nodes
            .iter()
            .map(|nd| {
                [
                    (nd.delta as f32).to_bits(),
                    nd.next,
                    nd.body_start,
                    nd.body_count,
                ]
            })
            .collect();

        let params = Params {
            n: n as u32,
            n_nodes: n_nodes as u32,
            eps2: (self.softening * self.softening) as f32,
            g: self.g as f32,
            theta: self.theta as f32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let bodies_buf = self.bodies_buf.as_ref().expect("buffers ensured above");
        let leaf_buf = self.leaf_buf.as_ref().expect("buffers ensured above");
        let accel_buf = self.accel_buf.as_ref().expect("buffers ensured above");
        let readback = self.readback_buf.as_ref().expect("buffers ensured above");
        let node_ch_buf = self.node_ch_buf.as_ref().expect("buffers ensured above");
        let node_cm_buf = self.node_cm_buf.as_ref().expect("buffers ensured above");
        let node_meta_buf = self.node_meta_buf.as_ref().expect("buffers ensured above");
        let bind_group = self.bind_group.as_ref().expect("bind group ensured above");

        self.queue
            .write_buffer(bodies_buf, 0, bytemuck::cast_slice(&bodies));
        self.queue
            .write_buffer(leaf_buf, 0, bytemuck::cast_slice(&flat.leaf_bodies));
        self.queue
            .write_buffer(node_ch_buf, 0, bytemuck::cast_slice(&node_ch));
        self.queue
            .write_buffer(node_cm_buf, 0, bytemuck::cast_slice(&node_cm));
        self.queue
            .write_buffer(node_meta_buf, 0, bytemuck::cast_slice(&node_meta));
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-tree-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(accel_buf, 0, readback, 0, bytes);
        self.queue.submit([enc.finish()]);

        // Map, block until the GPU is done, widen f32 accelerations back to f64. Like
        // the CPU solvers' assert on this infallible trait path, a map failure here is
        // an exceptional GPU loss and panics rather than silently corrupting state.
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

    /// Softened potential energy, delegated to the shared CPU **f64** reduction —
    /// identical to `GpuDirectSum`/`BarnesHut`. Same documented inconsistency: forces
    /// are f32 while this is f64, so an energy-drift diagnostic mixes a precision gap
    /// with integrator error; it is a periodic O(N²) diagnostic, not the per-step path.
    fn potential_energy(&self, state: &State) -> f64 {
        galaxy_solvers::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}
