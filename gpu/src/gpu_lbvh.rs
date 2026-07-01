//! [`GpuLbvh`]: GPU Linear-BVH Barnes-Hut force solver (DESIGN M4g) — the fifth and final
//! stage of the GPU-resident LBVH build, and the first that runs the **whole f32 pipeline
//! end-to-end**.
//!
//! Where [`crate::GpuTree`] builds+linearizes an octree on the CPU and only *traverses* on
//! the GPU, `GpuLbvh` runs the entire build on the GPU: `GpuMorton` (f32 codes) →
//! `GpuSort` (permutation) → gather → `GpuLbvhBuilder` (Karras tree + aggregate) →
//! `GpuLbvhFlattener` (DFS skip-pointer flat form) → an f32 stackless **traversal** kernel
//! that walks that flat form with a single node index. Same `(g, softening, theta)`
//! semantics and Plummer-softened kernel as [`galaxy_solvers::Lbvh`], so it is directly
//! comparable — an O(N log N) monopole approximation that reproduces direct summation as
//! `theta → 0`.
//!
//! ## θ→0 is where the end-to-end f32 topology straddle is finally checked
//! The f32 Morton stage can quantize a coordinate into a different 1024³ cell than the f64
//! reference (the M4c divergence), so the GPU's tree *topology* may differ from the CPU
//! `Lbvh`'s. θ→0 opens every node down to its leaves, so the walk *is* direct summation
//! **regardless of topology** — insensitive to that straddle, yet still catching any dropped
//! or double-counted subtree or bad skip pointer. So the θ→0 gate does not assert the
//! topology matches; it shows the f32 pipeline runs end-to-end and *still* yields exact
//! forces despite a possibly-different topology.
//!
//! ## Precision & determinism: same story as [`crate::GpuTree`]
//! The traversal is f32 (wgpu/naga has no portable f64 compute); the dominant error is the
//! same catastrophic cancellation in `xᵢ − xⱼ` and terms swallowed in the f32 accumulator,
//! worst in the large-coordinate collision regime. One invocation per target writes
//! `acc[i]` once from a private accumulator over a fixed skip-pointer order, and the whole
//! GPU build is float-`atomicAdd`-free (M4d/M4e/M4f are single-invocation or race-free), so
//! the result is bit-deterministic **on a given device**; cross-device equality is not
//! claimed.
//!
//! ## Scope: reference-grade composition, GPU-resident fuse deferred
//! Each stage owns its own wgpu device and the pointer tree / flat form round-trips through
//! host memory between stages (the M4d/M4e/M4f readback pattern), so a `GpuLbvh` holds
//! several devices and re-uploads between stages. That is the reference-grade build; the
//! **single-device, GPU-resident fuse** (no host round-trips, state kept on the GPU across
//! steps) is the named scale refinement — the same deferral every earlier stage carries.

use bytemuck::{Pod, Zeroable};

use galaxy_core::{DVec3, ForceSolver, State};

use crate::{GpuError, GpuLbvhFlattener, GpuMortonBuilder, GpuSorter};

/// Compute workgroup size. Tree traversal is irregular (no shared-memory tile), so 64 —
/// matching [`crate::GpuTree`] — is used rather than a wider tile.
const WORKGROUP_SIZE: u32 = 64;

/// Stackless **gather** traversal of the M4f flat LBVH. One invocation per target `i` walks
/// the flat form with a single `node` index: `mass ≤ 0` skips via `next`; a leaf
/// (`body_count > 0`) direct-sums its bodies (excluding self); an internal node is accepted
/// as a monopole when the Barnes (1994) criterion holds — over the binary node's **per-axis**
/// half-extents (cell size `s = 2·max(half)`, not [`crate::GpuTree`]'s scalar cube half) —
/// else opened to its first child (`node + 1`). A correct flatten makes `node` strictly
/// increase, so the walk terminates in ≤ `n_nodes` steps with no stack. Mirrors the CPU
/// [`galaxy_solvers::LbvhFlat::accel`] walk exactly.
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

@group(0) @binding(0) var<storage, read> bodies:      array<vec4<f32>>;   // xyz=pos, w=mass
@group(0) @binding(1) var<storage, read> node_center: array<vec4<f32>>;   // xyz=center, w=delta
@group(0) @binding(2) var<storage, read> node_half:   array<vec4<f32>>;   // xyz=half-extents
@group(0) @binding(3) var<storage, read> node_cm:     array<vec4<f32>>;   // xyz=com, w=mass
@group(0) @binding(4) var<storage, read> node_meta:   array<vec4<u32>>;   // x=next,y=body_start,z=body_count
@group(0) @binding(5) var<storage, read> leaf_bodies: array<u32>;
@group(0) @binding(6) var<storage, read_write> accel: array<vec4<f32>>;   // xyz=accel
@group(0) @binding(7) var<uniform> params: Params;

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
        let next = md.x;
        if (mass <= 0.0) { node = next; continue; }

        let bcount = md.z;
        if (bcount > 0u) {
            // Leaf: exact direct sum over its bodies, excluding the self term.
            let bstart = md.y;
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
            // Internal: never approximate a cell that contains the target. Cell size is the
            // AABB's longest side (a binary node is non-cubic — per-axis half-extents).
            let center = node_center[node].xyz;
            let delta = node_center[node].w;
            let half = node_half[node].xyz;
            let inside = all(abs(xp - center) <= half);
            let dx = cm.xyz - xp;
            let d2 = dot(dx, dx);
            let d = sqrt(d2);
            let s = 2.0 * max(half.x, max(half.y, half.z));
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

/// GPU Barnes-Hut force solver over a GPU-resident Morton Linear BVH. Same
/// `(g, softening, theta)` semantics as [`galaxy_solvers::Lbvh`], evaluated by the full GPU
/// f32 chain + an f32 stackless traversal.
pub struct GpuLbvh {
    /// Gravitational constant.
    g: f64,
    /// Plummer softening length ε.
    softening: f64,
    /// Opening angle θ.
    theta: f64,
    // The GPU build chain (each owns its own device — the reference-grade composition).
    morton: GpuMortonBuilder,
    sorter: GpuSorter,
    flattener: GpuLbvhFlattener,
    // The traversal device + pipeline.
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    // N-sized buffers (bodies, leaf indices, accel, readback) grow with particle count; node
    // buffers grow with the flat node count. Both grow lazily; the bind group is rebuilt
    // whenever any buffer is reallocated.
    bodies_buf: Option<wgpu::Buffer>,
    leaf_buf: Option<wgpu::Buffer>,
    accel_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,
    node_center_buf: Option<wgpu::Buffer>,
    node_half_buf: Option<wgpu::Buffer>,
    node_cm_buf: Option<wgpu::Buffer>,
    node_meta_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    body_capacity: usize,
    node_capacity: usize,
}

impl GpuLbvh {
    /// Bring up the GPU build chain and the traversal pipeline. Returns a typed
    /// [`GpuError`] (never panics) when no adapter is available.
    ///
    /// Requests **no** device features (baseline storage-buffer compute) and uses 7 storage
    /// buffers plus 1 uniform binding, within the default `maxStorageBuffersPerShaderStage`
    /// of 8 — so, like the other GPU stages, it does not narrow adapter support.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(g, softening, theta))
    }

    async fn new_async(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        let morton = GpuMortonBuilder::new()?;
        let sorter = GpuSorter::new()?;
        let flattener = GpuLbvhFlattener::new()?;

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
                label: Some("galaxy-gpu-lbvh-traverse-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-lbvh-traverse-shader"),
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
            label: Some("gpu-lbvh-traverse-bgl"),
            entries: &[
                storage(0, true),  // bodies
                storage(1, true),  // node center/delta
                storage(2, true),  // node half-extents
                storage(3, true),  // node com/mass
                storage(4, true),  // node meta
                storage(5, true),  // leaf bodies
                storage(6, false), // accel (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
            label: Some("gpu-lbvh-traverse-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-lbvh-traverse-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-lbvh-traverse-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuLbvh {
            g,
            softening,
            theta,
            morton,
            sorter,
            flattener,
            device,
            queue,
            pipeline,
            bgl,
            params_buf,
            bodies_buf: None,
            leaf_buf: None,
            accel_buf: None,
            readback_buf: None,
            node_center_buf: None,
            node_half_buf: None,
            node_cm_buf: None,
            node_meta_buf: None,
            bind_group: None,
            body_capacity: 0,
            node_capacity: 0,
        })
    }

    /// Ensure the N-sized and node-sized buffers hold at least `n` bodies and `n_nodes`
    /// nodes, (re)building the bind group when any buffer is reallocated. Only grows — a
    /// later smaller problem reuses the larger buffers (the kernel bounds itself by
    /// `params.n` / `params.n_nodes`). Caller guarantees `n > 0` and `n_nodes > 0`.
    fn ensure_capacity(&mut self, n: usize, n_nodes: usize) {
        let mut rebuild = self.bind_group.is_none();

        if n > self.body_capacity {
            let vec4_bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
            self.bodies_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-bodies"),
                size: vec4_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.leaf_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-leaf-bodies"),
                size: (n * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.accel_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-accel"),
                size: vec4_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.readback_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-readback"),
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
            self.node_center_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-node-center"),
                size: f4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_half_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-node-half"),
                size: f4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_cm_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-node-cm"),
                size: f4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_meta_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-lbvh-node-meta"),
                size: u4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.node_capacity = n_nodes;
            rebuild = true;
        }

        if rebuild {
            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu-lbvh-bind-group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.bodies_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.node_center_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.node_half_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.node_cm_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.node_meta_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.leaf_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.accel_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.params_buf.as_entire_binding(),
                    },
                ],
            }));
        }
    }
}

impl ForceSolver for GpuLbvh {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        if n == 1 {
            // A lone particle feels no force (its only leaf holds just itself). Avoids
            // running the whole build chain on a single point.
            acc[0] = DVec3::ZERO;
            return;
        }

        // --- The full GPU f32 build chain (chain A): the end-to-end straddle. ---
        // f32 Morton codes → GPU stable sort → gather leaf payload → GPU Karras build +
        // aggregate → GPU DFS skip-pointer flatten.
        let codes = self.morton.compute(&state.pos).codes;
        let sort = self.sorter.sort(&codes);
        let order = sort.order;
        let sorted_codes = sort.sorted_codes;
        let sorted_pos: Vec<DVec3> = order.iter().map(|&i| state.pos[i as usize]).collect();
        let sorted_mass: Vec<f64> = order.iter().map(|&i| state.mass[i as usize]).collect();
        let flat = self
            .flattener
            .build_flat(&sorted_codes, &sorted_pos, &sorted_mass, &order);

        let n_nodes = flat.next.len(); // == 2N-1
        self.ensure_capacity(n, n_nodes);

        // Pack the traversal inputs. Bodies stay in ORIGINAL order (the leaf_bodies hold
        // original indices; the kernel excludes the self term by original index). The
        // f64→f32 narrowing here is the precision reduction the crate docs own.
        let bodies: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let p = state.pos[i];
                [p.x as f32, p.y as f32, p.z as f32, state.mass[i] as f32]
            })
            .collect();
        let node_center: Vec<[f32; 4]> = (0..n_nodes)
            .map(|d| {
                let c = flat.center[d];
                [c[0], c[1], c[2], flat.delta[d]]
            })
            .collect();
        let node_half: Vec<[f32; 4]> = (0..n_nodes)
            .map(|d| {
                let h = flat.half_extents[d];
                [h[0], h[1], h[2], 0.0]
            })
            .collect();
        let node_cm: Vec<[f32; 4]> = (0..n_nodes)
            .map(|d| {
                let c = flat.com[d];
                [c[0], c[1], c[2], flat.mass[d]]
            })
            .collect();
        let node_meta: Vec<[u32; 4]> = (0..n_nodes)
            .map(|d| [flat.next[d], flat.body_start[d], flat.body_count[d], 0])
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
        let node_center_buf = self
            .node_center_buf
            .as_ref()
            .expect("buffers ensured above");
        let node_half_buf = self.node_half_buf.as_ref().expect("buffers ensured above");
        let node_cm_buf = self.node_cm_buf.as_ref().expect("buffers ensured above");
        let node_meta_buf = self.node_meta_buf.as_ref().expect("buffers ensured above");
        let bind_group = self.bind_group.as_ref().expect("bind group ensured above");

        self.queue
            .write_buffer(bodies_buf, 0, bytemuck::cast_slice(&bodies));
        self.queue
            .write_buffer(leaf_buf, 0, bytemuck::cast_slice(&flat.leaf_bodies));
        self.queue
            .write_buffer(node_center_buf, 0, bytemuck::cast_slice(&node_center));
        self.queue
            .write_buffer(node_half_buf, 0, bytemuck::cast_slice(&node_half));
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
                label: Some("gpu-lbvh-traverse-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(accel_buf, 0, readback, 0, bytes);
        self.queue.submit([enc.finish()]);

        // Map, block until the GPU is done, widen f32 accelerations back to f64. A map
        // failure here is an exceptional GPU loss and panics rather than corrupt state.
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

    /// Softened potential energy, delegated to the shared CPU **f64** reduction — identical
    /// to `GpuTree`/`BarnesHut`. Same documented inconsistency: forces are f32 while this is
    /// f64, so an energy-drift diagnostic mixes a precision gap with integrator error; it is
    /// a periodic O(N²) diagnostic, not the per-step path.
    fn potential_energy(&self, state: &State) -> f64 {
        galaxy_solvers::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}
