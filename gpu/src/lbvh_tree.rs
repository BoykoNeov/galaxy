//! [`GpuLbvhBuilder`]: the GPU Karras tree-build kernel + atomic-flag bottom-up
//! aggregation — stage 3 of the GPU-resident LBVH build (DESIGN M4e).
//!
//! Given the Morton `sorted_codes` (the output of [`crate::GpuSorter`] / the CPU
//! [`galaxy_solvers::reference_sort`]) plus the per-leaf position + mass **pre-gathered
//! into that sorted order**, it builds — on the GPU (wgpu compute) — the binary radix
//! tree of [`galaxy_solvers::reference_karras`] (topology) and its bottom-up
//! [`galaxy_solvers::reference_aggregate`] (per-node AABB / centre-of-mass / mass).
//!
//! ## Two gates, because the stage is half integer and half f32
//! The Karras **topology** is a pure-integer function of the sorted codes
//! (`δ = clz(code_a ^ code_b)`, with a position tie-extension for equal codes), so the
//! GPU `(left, right, parent)` must equal the CPU reference **bit-for-bit** — the
//! load-bearing gate, exactly like the M4d sort. The **aggregation** runs in f32: its
//! AABB `min`/`max` folds never round and are order-independent (bit-exact vs an f32 CPU
//! fold), while `com`/`mass` are f32-lossy and gated on tolerance vs the f64 reference.
//!
//! This does **not** contradict the M4c f32-divergence note: that divergence lives
//! upstream in the Morton *codes*, not in this pure-integer tree step. Fed reference /
//! GPU-sorted codes (bit-identical per M4d), the topology is bit-exact.
//!
//! ## Two kernels: parallel topology, single-invocation aggregation
//! **`build_tree`** is genuinely parallel (one invocation per internal node): each writes
//! only its own `left`/`right` and its two children's `parent` slot, and reads no other
//! invocation's output — race-free, no atomics. Its δ search runs in **signed `i32`**:
//! `delta` returns −1 for out-of-range probes, so a `u32` port would treat −1 as
//! `0xFFFFFFFF` and win every boundary comparison (the #1 silent-corruption trap).
//!
//! **`aggregate`** is a **single invocation** (`@workgroup_size(1)`, dispatched `(1,1,1)`).
//! The parallel Karras atomic-flag walk needs a device-scope memory fence to publish a
//! sibling's non-atomic AABB writes across workgroups — WGSL 1.0 has none (`storageBarrier`
//! is workgroup-only), so it is unsound to depend on. Collapsing to one serial invocation
//! makes every read trivially visible: correctness is unarguable and determinism is free
//! (the same discipline as the M4d single-invocation sort). The counter is still the
//! Karras visit-**flag** — a node folds only when its *second* child arrives (both
//! subtrees final), from its **stored** `left`/`right` in fixed order, so the result is
//! order-independent. The parallel atomic-flag build (with device fences) is the named
//! scale refinement, alongside keeping state GPU-resident.
//!
//! ## Scope: the raw pointer tree, flatten deferred
//! This stage emits the raw pointer-based tree (parent + children + raw
//! `min`/`max`/`com`/`mass`), **not** the DFS skip-pointer `LbvhFlat` form — deriving
//! `center`/`half`/`delta` and the `next` skip pointer (a subtree-size prefix-sum /
//! Euler-tour) is the next stage, which lets the deferred `GpuLbvh` traverse the same
//! form the CPU `LbvhFlat::accel` walk uses.

use bytemuck::{Pod, Zeroable};
use galaxy_core::DVec3;
use galaxy_solvers::NO_PARENT;

use crate::GpuError;

/// Threads per workgroup for the parallel tree-build pass.
const WORKGROUP_SIZE: u32 = 256;

/// The GPU-built Karras binary radix tree in the canonical unified node layout
/// ([`galaxy_solvers::KarrasTree`]): the `N` leaves occupy unified indices `[0, N)` (in
/// Morton-sorted order), the `N-1` internal nodes `[N, 2N-1)` (internal `i` at `N+i`;
/// root = internal 0 = unified `N`). A unified index `u` is a **leaf iff `u < N`**.
pub struct GpuLbvhTree {
    /// Number of leaves `N`.
    pub n: usize,
    /// Left child (unified index) of each of the `N-1` internal nodes.
    pub left: Vec<u32>,
    /// Right child (unified index) of each of the `N-1` internal nodes.
    pub right: Vec<u32>,
    /// Parent (unified index) of every node (`len 2N-1`); the root is
    /// [`galaxy_solvers::NO_PARENT`].
    pub parent: Vec<u32>,
    /// AABB low corner per unified node (`len 2N-1`, f32).
    pub aabb_min: Vec<[f32; 3]>,
    /// AABB high corner per unified node (`len 2N-1`, f32).
    pub aabb_max: Vec<[f32; 3]>,
    /// Aggregate centre of mass per unified node (`len 2N-1`, f32).
    pub com: Vec<[f32; 3]>,
    /// Aggregate mass per unified node (`len 2N-1`, f32).
    pub mass: Vec<f32>,
}

/// Uniform block mirroring the WGSL `Params` (16-byte aligned): the leaf count `N`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad: [u32; 3],
}

/// Karras tree-build (parallel) + atomic-flag bottom-up aggregation (single invocation).
/// One module, two entry points sharing the bind group — codes/`children`/`parent` for
/// the build; `leaf`/`node_*`/`counter` for the aggregation. `pub(crate)` so the M4h fuse
/// runs the same `build_tree`/`aggregate` kernels.
pub(crate) const SHADER: &str = r#"
const NO_PARENT: u32 = 0xffffffffu;

struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       codes:    array<u32>;
@group(0) @binding(2) var<storage, read>       leaf:     array<vec4<f32>>;  // xyz=pos, w=mass
// Children interleaved (left at 2i, right at 2i+1) — one buffer, to stay within the
// baseline 8-storage-buffer limit (no adapter narrowing).
@group(0) @binding(3) var<storage, read_write> children: array<u32>;
@group(0) @binding(4) var<storage, read_write> parent:   array<u32>;
@group(0) @binding(5) var<storage, read_write> node_min: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> node_max: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> node_com: array<vec4<f32>>;  // xyz=com, w=mass
@group(0) @binding(8) var<storage, read_write> counter:  array<atomic<u32>>;

// Augmented-key common-prefix length, SIGNED. delta returns -1 for out-of-range probes,
// so the whole search runs in i32 (a u32 port would treat -1 as 0xFFFFFFFF and win every
// boundary comparison). Mirrors galaxy_solvers::lbvh::delta: on a code tie, extend into
// the distinct sorted positions (32 + clz(a ^ b)); countLeadingZeros(0u) == 32u matches
// Rust u32::leading_zeros.
fn delta(a: i32, b: i32) -> i32 {
    let n = i32(params.n);
    if (b < 0 || b >= n) { return -1; }
    let ca = codes[a];
    let cb = codes[b];
    if (ca == cb) {
        return 32 + i32(countLeadingZeros(u32(a) ^ u32(b)));
    }
    return i32(countLeadingZeros(ca ^ cb));
}

// One Karras internal node per invocation (Karras 2012, Algorithms 3-4). Race-free: writes
// only children[2i]/children[2i+1] and its two children's parent slot.
@compute @workgroup_size(256)
fn build_tree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let ii = gid.x;
    if (ii >= n - 1u) { return; }
    let i = i32(ii);

    // determineRange: direction toward the neighbour sharing the longer prefix.
    var dir: i32 = -1;
    if (delta(i, i + 1) > delta(i, i - 1)) { dir = 1; }
    let delta_min = delta(i, i - dir);

    // exponential upper bound on the range length, then binary search for the far end.
    var l_max: i32 = 2;
    while (delta(i, i + l_max * dir) > delta_min) { l_max = l_max * 2; }
    var l: i32 = 0;
    var t: i32 = l_max / 2;
    while (t > 0) {
        if (delta(i, i + (l + t) * dir) > delta_min) { l = l + t; }
        t = t / 2;
    }
    let j = i + l * dir;

    // binary search for the split, by delta over the range's own prefix.
    let delta_node = delta(i, j);
    var s: i32 = 0;
    var ts: i32 = l;
    loop {
        ts = (ts + 1) / 2;               // ceil-halve
        if (delta(i, i + (s + ts) * dir) > delta_node) { s = s + ts; }
        if (ts <= 1) { break; }
    }
    var gamma = i + s * dir;
    if (dir < 0) { gamma = gamma - 1; }  // dir.min(0)

    let lo = min(i, j);
    let hi = max(i, j);
    var left_u: u32;
    if (lo == gamma)     { left_u  = u32(gamma); }     else { left_u  = n + u32(gamma); }
    var right_u: u32;
    if (hi == gamma + 1) { right_u = u32(gamma + 1); } else { right_u = n + u32(gamma + 1); }

    children[2u * ii]      = left_u;
    children[2u * ii + 1u] = right_u;
    let me = n + ii;
    parent[left_u]  = me;
    parent[right_u] = me;
}

// Fold two children into their parent in fixed (left, right) order — the exact combine of
// galaxy_solvers::lbvh::fold_agg (mass sum, mass-weighted com with a geometric-midpoint
// fallback, AABB union).
fn fold(node: u32, l: u32, r: u32) {
    let lmin = node_min[l].xyz;
    let lmax = node_max[l].xyz;
    let rmin = node_min[r].xyz;
    let rmax = node_max[r].xyz;
    let lc = node_com[l];
    let rc = node_com[r];
    let m = lc.w + rc.w;
    var com: vec3<f32>;
    if (m > 0.0) {
        com = (lc.xyz * lc.w + rc.xyz * rc.w) / m;
    } else {
        com = (lmin + lmax + rmin + rmax) * 0.25;
    }
    node_min[node] = vec4<f32>(min(lmin, rmin), 0.0);
    node_max[node] = vec4<f32>(max(lmax, rmax), 0.0);
    node_com[node] = vec4<f32>(com, m);
}

// Single-invocation serial bottom-up aggregation (see the module doc for why one
// invocation). Seed every leaf, then walk each leaf up: the SECOND child to reach a node
// folds it (both subtrees final). The counter must be zeroed by the host before dispatch.
@compute @workgroup_size(1)
fn aggregate() {
    let n = params.n;
    for (var k = 0u; k < n; k = k + 1u) {
        let lp = leaf[k];
        node_min[k] = vec4<f32>(lp.xyz, 0.0);
        node_max[k] = vec4<f32>(lp.xyz, 0.0);
        node_com[k] = vec4<f32>(lp.xyz, lp.w);
    }
    let cap = 2u * n; // safety bound comfortably >= max depth (N-1 for a degenerate chain)
    for (var k = 0u; k < n; k = k + 1u) {
        var node = parent[k];
        var step = 0u;
        loop {
            if (node == NO_PARENT || step >= cap) { break; }
            step = step + 1u;
            let ci = node - n;
            let prev = atomicAdd(&counter[ci], 1u);
            if (prev == 0u) { break; }         // first child; the second will fold
            fold(node, children[2u * ci], children[2u * ci + 1u]); // both subtrees final
            node = parent[node];
        }
    }
}
"#;

/// GPU Karras tree-build + atomic-flag aggregation stage. Holds a reusable wgpu compute
/// context built once and driven per [`build`](Self::build) call; storage buffers grow
/// lazily with N. Same bring-up idiom as [`crate::GpuSorter`].
pub struct GpuLbvhBuilder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    build_pipeline: wgpu::ComputePipeline,
    aggregate_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    // Storage grown lazily to the largest N seen; the bind group is rebuilt on (re)alloc.
    codes_buf: Option<wgpu::Buffer>,
    leaf_buf: Option<wgpu::Buffer>,
    children_buf: Option<wgpu::Buffer>,
    parent_buf: Option<wgpu::Buffer>,
    node_min_buf: Option<wgpu::Buffer>,
    node_max_buf: Option<wgpu::Buffer>,
    node_com_buf: Option<wgpu::Buffer>,
    counter_buf: Option<wgpu::Buffer>,
    children_rb: Option<wgpu::Buffer>,
    parent_rb: Option<wgpu::Buffer>,
    node_min_rb: Option<wgpu::Buffer>,
    node_max_rb: Option<wgpu::Buffer>,
    node_com_rb: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    capacity: usize,
}

impl GpuLbvhBuilder {
    /// Bring up a headless wgpu compute device and build the tree-build + aggregation
    /// pipelines. Returns a typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let crate::context::GpuContext { device, queue } = crate::context::gpu_context()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-lbvh-tree-shader"),
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
            label: Some("gpu-lbvh-tree-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),  // codes    (read)
                storage(2, true),  // leaf     (read)
                storage(3, false), // children (rw, interleaved left/right)
                storage(4, false), // parent   (rw)
                storage(5, false), // node_min (rw)
                storage(6, false), // node_max (rw)
                storage(7, false), // node_com (rw)
                storage(8, false), // counter  (rw, atomic)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-lbvh-tree-pl"),
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
        let build_pipeline = make_pipeline("build_tree", "gpu-lbvh-build");
        let aggregate_pipeline = make_pipeline("aggregate", "gpu-lbvh-aggregate");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-lbvh-tree-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuLbvhBuilder {
            device,
            queue,
            build_pipeline,
            aggregate_pipeline,
            bgl,
            params_buf,
            codes_buf: None,
            leaf_buf: None,
            children_buf: None,
            parent_buf: None,
            node_min_buf: None,
            node_max_buf: None,
            node_com_buf: None,
            counter_buf: None,
            children_rb: None,
            parent_rb: None,
            node_min_rb: None,
            node_max_rb: None,
            node_com_rb: None,
            bind_group: None,
            capacity: 0,
        })
    }

    /// Ensure the storage / readback buffers hold at least `n` leaves, rebuilding the bind
    /// group when they are (re)allocated. Only grows. Caller guarantees `n >= 2`.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity && self.bind_group.is_some() {
            return;
        }
        let total = 2 * n - 1;
        let u32_n = (n * 4) as u64;
        let vec4_n = (n * 16) as u64;
        let u32_children = ((n - 1) * 2 * 4) as u64; // interleaved left/right
        let u32_nm1 = ((n - 1) * 4) as u64;
        let u32_total = (total * 4) as u64;
        let vec4_total = (total * 16) as u64;

        let dev = &self.device;
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let cdst = wgpu::BufferUsages::COPY_DST;
        let csrc = wgpu::BufferUsages::COPY_SRC;
        let mapread = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        let codes_buf = make("gpu-lbvh-codes", u32_n, storage | cdst);
        let leaf_buf = make("gpu-lbvh-leaf", vec4_n, storage | cdst);
        let children_buf = make("gpu-lbvh-children", u32_children, storage | csrc);
        let parent_buf = make("gpu-lbvh-parent", u32_total, storage | cdst | csrc);
        let node_min_buf = make("gpu-lbvh-node-min", vec4_total, storage | csrc);
        let node_max_buf = make("gpu-lbvh-node-max", vec4_total, storage | csrc);
        let node_com_buf = make("gpu-lbvh-node-com", vec4_total, storage | csrc);
        let counter_buf = make("gpu-lbvh-counter", u32_nm1, storage | cdst);

        let children_rb = make("gpu-lbvh-children-rb", u32_children, mapread);
        let parent_rb = make("gpu-lbvh-parent-rb", u32_total, mapread);
        let node_min_rb = make("gpu-lbvh-node-min-rb", vec4_total, mapread);
        let node_max_rb = make("gpu-lbvh-node-max-rb", vec4_total, mapread);
        let node_com_rb = make("gpu-lbvh-node-com-rb", vec4_total, mapread);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-lbvh-tree-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: codes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: leaf_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: children_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: parent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: node_min_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: node_max_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: node_com_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: counter_buf.as_entire_binding(),
                },
            ],
        });

        self.codes_buf = Some(codes_buf);
        self.leaf_buf = Some(leaf_buf);
        self.children_buf = Some(children_buf);
        self.parent_buf = Some(parent_buf);
        self.node_min_buf = Some(node_min_buf);
        self.node_max_buf = Some(node_max_buf);
        self.node_com_buf = Some(node_com_buf);
        self.counter_buf = Some(counter_buf);
        self.children_rb = Some(children_rb);
        self.parent_rb = Some(parent_rb);
        self.node_min_rb = Some(node_min_rb);
        self.node_max_rb = Some(node_max_rb);
        self.node_com_rb = Some(node_com_rb);
        self.bind_group = Some(bind_group);
        self.capacity = n;
    }

    /// Build the Karras tree over the **sorted** `sorted_codes` (as produced by the GPU
    /// sort), with `sorted_pos`/`sorted_mass` the leaf payload gathered into the same
    /// sorted order (leaf `k` → `sorted_pos[k]`). `sorted_codes` may be empty (yields an
    /// empty tree). Panics if the three input lengths disagree.
    pub fn build(
        &mut self,
        sorted_codes: &[u32],
        sorted_pos: &[DVec3],
        sorted_mass: &[f64],
    ) -> GpuLbvhTree {
        let n = sorted_codes.len();
        assert_eq!(sorted_pos.len(), n, "sorted_pos length must equal N");
        assert_eq!(sorted_mass.len(), n, "sorted_mass length must equal N");

        if n == 0 {
            return GpuLbvhTree {
                n: 0,
                left: Vec::new(),
                right: Vec::new(),
                parent: Vec::new(),
                aabb_min: Vec::new(),
                aabb_max: Vec::new(),
                com: Vec::new(),
                mass: Vec::new(),
            };
        }
        if n == 1 {
            // A single leaf is the whole tree — no dispatch (matches the solver convention).
            let p = [
                sorted_pos[0].x as f32,
                sorted_pos[0].y as f32,
                sorted_pos[0].z as f32,
            ];
            return GpuLbvhTree {
                n: 1,
                left: Vec::new(),
                right: Vec::new(),
                parent: vec![NO_PARENT],
                aabb_min: vec![p],
                aabb_max: vec![p],
                com: vec![p],
                mass: vec![sorted_mass[0] as f32],
            };
        }

        self.ensure_capacity(n);
        let total = 2 * n - 1;

        // Upload params + codes + the packed leaf payload (xyz = position, w = mass, f32).
        let params = Params {
            n: n as u32,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let codes_buf = self.codes_buf.as_ref().expect("buffers ensured above");
        self.queue
            .write_buffer(codes_buf, 0, bytemuck::cast_slice(sorted_codes));
        let leaf_data: Vec<[f32; 4]> = sorted_pos
            .iter()
            .zip(sorted_mass)
            .map(|(p, &m)| [p.x as f32, p.y as f32, p.z as f32, m as f32])
            .collect();
        let leaf_buf = self.leaf_buf.as_ref().expect("buffers ensured above");
        self.queue
            .write_buffer(leaf_buf, 0, bytemuck::cast_slice(&leaf_data));

        // Initialize every parent to NO_PARENT (0xFFFFFFFF): build_tree overwrites every
        // non-root slot, and this leaves the root's as the aggregation's stop sentinel.
        // (A zeroing clear_buffer would be wrong — NO_PARENT is all-ones, not zero.)
        let parent_init = vec![NO_PARENT; total];
        let parent_buf = self.parent_buf.as_ref().expect("buffers ensured above");
        self.queue
            .write_buffer(parent_buf, 0, bytemuck::cast_slice(&parent_init));

        let bind_group = self.bind_group.as_ref().expect("bind group ensured above");
        let counter_buf = self.counter_buf.as_ref().expect("buffers ensured above");

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Parallel Karras tree-build: one invocation per internal node.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-lbvh-build-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((n as u32 - 1).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        // Zero the visit counter, then the single-invocation bottom-up aggregation.
        enc.clear_buffer(counter_buf, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-lbvh-aggregate-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.aggregate_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single invocation (see module doc)
        }

        // Copy the results to the mappable readback buffers.
        let children_buf = self.children_buf.as_ref().unwrap();
        let node_min_buf = self.node_min_buf.as_ref().unwrap();
        let node_max_buf = self.node_max_buf.as_ref().unwrap();
        let node_com_buf = self.node_com_buf.as_ref().unwrap();
        let children_rb = self.children_rb.as_ref().unwrap();
        let parent_rb = self.parent_rb.as_ref().unwrap();
        let node_min_rb = self.node_min_rb.as_ref().unwrap();
        let node_max_rb = self.node_max_rb.as_ref().unwrap();
        let node_com_rb = self.node_com_rb.as_ref().unwrap();
        let u32_children = ((n - 1) * 2 * 4) as u64;
        let u32_total = (total * 4) as u64;
        let vec4_total = (total * 16) as u64;
        enc.copy_buffer_to_buffer(children_buf, 0, children_rb, 0, u32_children);
        enc.copy_buffer_to_buffer(parent_buf, 0, parent_rb, 0, u32_total);
        enc.copy_buffer_to_buffer(node_min_buf, 0, node_min_rb, 0, vec4_total);
        enc.copy_buffer_to_buffer(node_max_buf, 0, node_max_rb, 0, vec4_total);
        enc.copy_buffer_to_buffer(node_com_buf, 0, node_com_rb, 0, vec4_total);
        self.queue.submit([enc.finish()]);

        // Map all five readbacks, wait once, read. A map failure is a genuine GPU loss
        // (new() validated the device), so it panics rather than corrupt state — matching
        // the other GPU stages.
        let slices = [
            children_rb.slice(..u32_children),
            parent_rb.slice(..u32_total),
            node_min_rb.slice(..vec4_total),
            node_max_rb.slice(..vec4_total),
            node_com_rb.slice(..vec4_total),
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        for slice in &slices {
            let tx = tx.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        for _ in 0..slices.len() {
            rx.recv()
                .expect("map channel closed")
                .expect("gpu buffer map failed");
        }

        // Split the interleaved children (left at 2i, right at 2i+1) back into two arrays.
        let children_data = slices[0].get_mapped_range();
        let ch: &[u32] = bytemuck::cast_slice(&children_data);
        let left: Vec<u32> = (0..n - 1).map(|i| ch[2 * i]).collect();
        let right: Vec<u32> = (0..n - 1).map(|i| ch[2 * i + 1]).collect();
        drop(children_data);
        let parent: Vec<u32> =
            bytemuck::cast_slice::<u8, u32>(&slices[1].get_mapped_range())[..total].to_vec();
        let unpack3 = |slice: &wgpu::BufferSlice| -> Vec<[f32; 3]> {
            let data = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&data);
            (0..total)
                .map(|u| [f[u * 4], f[u * 4 + 1], f[u * 4 + 2]])
                .collect()
        };
        let aabb_min = unpack3(&slices[2]);
        let aabb_max = unpack3(&slices[3]);
        let com_data = slices[4].get_mapped_range();
        let com_f: &[f32] = bytemuck::cast_slice(&com_data);
        let com: Vec<[f32; 3]> = (0..total)
            .map(|u| [com_f[u * 4], com_f[u * 4 + 1], com_f[u * 4 + 2]])
            .collect();
        let mass: Vec<f32> = (0..total).map(|u| com_f[u * 4 + 3]).collect();
        drop(com_data);

        children_rb.unmap();
        parent_rb.unmap();
        node_min_rb.unmap();
        node_max_rb.unmap();
        node_com_rb.unmap();

        GpuLbvhTree {
            n,
            left,
            right,
            parent,
            aabb_min,
            aabb_max,
            com,
            mass,
        }
    }
}
