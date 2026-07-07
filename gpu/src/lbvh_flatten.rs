//! [`GpuLbvhFlattener`]: the GPU DFS skip-pointer flatten — stage 4 of the GPU-resident
//! LBVH build (DESIGN M4f).
//!
//! Given the M4e Karras **pointer tree** (per unified node: two children, parent, and the
//! aggregated AABB `min`/`max` + com + mass), it linearizes the tree — on the GPU (wgpu
//! compute) — into the **DFS pre-order skip-pointer** form of [`galaxy_solvers::LbvhFlat`]:
//! a `2N-1`-slot array where slot 0 is the root, a node's first child is `slot+1`, and its
//! `next` skip pointer is one past its whole subtree. That stackless form is what the
//! deferred `GpuLbvh` traversal (M4g) walks with a single index — the exact shape the CPU
//! [`galaxy_solvers::LbvhFlat::accel`] walk uses.
//!
//! ## Two gates, split like M4e's build/aggregate
//! The **structure** (`next` / `body_start` / `body_count` and the `leaf_bodies`
//! permutation) is a pure-integer function of the topology — the DFS pre-order of a fixed
//! tree is deterministic — so it must equal [`galaxy_solvers::reference_flatten`]
//! **bit-for-bit**. The **geometry** (`center`/`half_extents` = `(min±max)/2`, `com`,
//! `mass`, and `delta = |com − center|`) runs in f32: min/max are exact under widening but
//! the halving sum rounds, com/mass are f32-lossy folds, and delta is an f32-lossy sqrt, so
//! all are gated on tolerance vs the f64 reference over the same narrowed leaves.
//!
//! ## Two kernels: single-invocation structure, parallel geometry
//! **`flatten_structure`** is a **single invocation** (`@workgroup_size(1)`), for the exact
//! reason M4e's aggregation is: the flatten is inherently serial (a DFS emission order) and
//! WGSL 1.0 has no device-scope fence to coordinate a parallel Euler-tour across
//! workgroups, so collapsing to one serial invocation makes correctness unarguable and
//! determinism free. It runs the DESIGN's **subtree-size prefix** in two passes over the
//! same buffers — *no recursion, no fixed-size stack*: (A) a bottom-up visit-flag climb
//! (the M4e aggregate shape) computes `size[u]`; (B) a top-down pre-order walk assigns each
//! node its DFS slot `d` (the emit counter) and writes `next = d + size[u]`. Pass B uses an
//! explicit stack **in a storage buffer** of capacity `2N-1` — a workgroup-local array
//! would overflow on a degenerate chain (depth `N-1`), the trap the `monotone_chain` gate
//! guards. `leaf_bodies[body_start] = order[u]` maps each sorted leaf back to its original
//! particle index (the space the traversal excludes the self term in).
//!
//! **`flatten_geometry`** is genuinely parallel (one invocation per DFS slot): it gathers
//! the node at `slot d` (its unified index is handed over in the structure pass's output)
//! and derives `center`/`half`/`com`/`mass`/`delta` from the M4e aggregate — race-free, no
//! atomics. The parallel Euler-tour flatten (which would fold both passes into one parallel
//! kernel) is the named scale refinement, alongside keeping the whole build GPU-resident
//! (this stage re-uploads the M4e pointer tree, mirroring the M4d/M4e readback pattern).

use bytemuck::{Pod, Zeroable};
use galaxy_core::DVec3;

use crate::{GpuError, GpuLbvhBuilder};

/// Threads per workgroup for the parallel geometry-gather pass.
const WORKGROUP_SIZE: u32 = 256;

/// The GPU-flattened LBVH in DFS pre-order with skip pointers — the SoA mirror of
/// [`galaxy_solvers::LbvhFlat`] the deferred `GpuLbvh` traversal walks. Every per-node
/// field has length `2N-1` (DFS slot order, root at slot 0); `leaf_bodies` has length `N`.
pub struct GpuLbvhFlat {
    /// Number of leaves `N`.
    pub n: usize,
    /// AABB geometric center per DFS slot (f32).
    pub center: Vec<[f32; 3]>,
    /// AABB half-extents per DFS slot (per axis, f32).
    pub half_extents: Vec<[f32; 3]>,
    /// Aggregate centre of mass per DFS slot (f32).
    pub com: Vec<[f32; 3]>,
    /// Aggregate mass per DFS slot (f32).
    pub mass: Vec<f32>,
    /// `|com − center|` per DFS slot (Barnes 1994 opening correction, f32).
    pub delta: Vec<f32>,
    /// Skip pointer per DFS slot: one past this node's subtree in pre-order.
    pub next: Vec<u32>,
    /// Leaf start offset into `leaf_bodies` (0 for internal nodes).
    pub body_start: Vec<u32>,
    /// Leaf body count — `> 0` iff leaf (always 1 for an LBVH leaf).
    pub body_count: Vec<u32>,
    /// Concatenated leaf body **original** particle indices, in DFS-leaf order.
    pub leaf_bodies: Vec<u32>,
}

impl GpuLbvhFlat {
    /// The empty (N=0) flat form.
    fn empty() -> Self {
        GpuLbvhFlat {
            n: 0,
            center: Vec::new(),
            half_extents: Vec::new(),
            com: Vec::new(),
            mass: Vec::new(),
            delta: Vec::new(),
            next: Vec::new(),
            body_start: Vec::new(),
            body_count: Vec::new(),
            leaf_bodies: Vec::new(),
        }
    }
}

/// Uniform block mirroring the WGSL `Params` (16-byte aligned): the leaf count `N`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad: [u32; 3],
}

/// Single-invocation DFS structure pass: subtree sizes (bottom-up), then the pre-order
/// slot/skip-pointer assignment (top-down). Emits `slot_meta` = `[next, body_start, body_count,
/// unified_index]` per DFS slot plus the `leaf_bodies` permutation. `size`/`stack` are
/// scratch (`stack` doubles as the per-internal visit flag in pass A, then the explicit DFS
/// stack in pass B). `pub(crate)` so the M4h fuse runs the same `flatten_structure` kernel
/// (the fuse writes its own geometry kernel in the traversal's packing, so `GEOMETRY_SHADER`
/// stays private).
pub(crate) const STRUCTURE_SHADER: &str = r#"
const NO_PARENT: u32 = 0xffffffffu;

struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read>       children:    array<u32>;   // interleaved L,R
@group(0) @binding(2) var<storage, read>       parent:      array<u32>;   // per unified node
@group(0) @binding(3) var<storage, read>       order:       array<u32>;   // sorted leaf -> orig
@group(0) @binding(4) var<storage, read_write> slot_meta:        array<vec4<u32>>; // per DFS slot
@group(0) @binding(5) var<storage, read_write> leaf_bodies: array<u32>;
@group(0) @binding(6) var<storage, read_write> size:        array<u32>;   // scratch, per unified
@group(0) @binding(7) var<storage, read_write> stack:       array<u32>;   // scratch

@compute @workgroup_size(1)
fn flatten_structure() {
    let n = params.n;

    // ---- Pass A: subtree sizes, bottom-up (reuse `stack` as per-internal visit flags).
    // Each internal node gets exactly two arrivals (one per child subtree, the M4e
    // aggregate invariant); the SECOND arrival folds size = 1 + size(L) + size(R), by
    // which time both child sizes are final.
    for (var c = 0u; c < n - 1u; c = c + 1u) { stack[c] = 0u; }
    for (var k = 0u; k < n; k = k + 1u) { size[k] = 1u; }
    let cap = 2u * n;                                    // safety bound >= max depth
    for (var k = 0u; k < n; k = k + 1u) {
        var node = parent[k];
        var step = 0u;
        loop {
            if (node == NO_PARENT || step >= cap) { break; }
            step = step + 1u;
            let ci = node - n;
            let prev = stack[ci];
            stack[ci] = prev + 1u;
            if (prev == 0u) { break; }                  // first child; second folds
            let l = children[2u * ci];
            let r = children[2u * ci + 1u];
            size[node] = 1u + size[l] + size[r];
            node = parent[node];
        }
    }

    // ---- Pass B: top-down pre-order DFS. `stack` is now an explicit node stack (capacity
    // 2N-1 >= max frontier). Slot d = emit counter (== DFS pre-order index); next = d+size.
    var sp = 0u;
    stack[0] = n;                                        // root = internal 0 = unified n
    sp = 1u;
    var emit = 0u;
    var leaf_ct = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let u = stack[sp];
        let d = emit;
        emit = emit + 1u;
        let su = size[u];
        if (u < n) {
            // Leaf (su == 1): next = d+1, one body, original index via `order`.
            slot_meta[d] = vec4<u32>(d + su, leaf_ct, 1u, u);
            leaf_bodies[leaf_ct] = order[u];
            leaf_ct = leaf_ct + 1u;
        } else {
            let ci = u - n;
            slot_meta[d] = vec4<u32>(d + su, 0u, 0u, u);
            // Push right then left so the left subtree is emitted first (pre-order).
            stack[sp] = children[2u * ci + 1u];
            sp = sp + 1u;
            stack[sp] = children[2u * ci];
            sp = sp + 1u;
        }
    }
}
"#;

/// Parallel geometry-gather pass: one invocation per DFS slot derives the flat node's
/// `center`/`half`/`com`/`mass`/`delta` from the M4e aggregate at the slot's unified index
/// (`slot_meta[d].w`). Race-free (each writes only its own slot); f32 (the tolerance gate).
const GEOMETRY_SHADER: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       slot_meta:     array<vec4<u32>>; // per DFS slot
@group(0) @binding(2) var<storage, read>       node_min: array<vec4<f32>>; // per unified node
@group(0) @binding(3) var<storage, read>       node_max: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       node_com: array<vec4<f32>>; // xyz=com, w=mass
@group(0) @binding(5) var<storage, read_write> geo0:     array<vec4<f32>>; // center.xyz, delta
@group(0) @binding(6) var<storage, read_write> geo1:     array<vec4<f32>>; // half.xyz,   mass
@group(0) @binding(7) var<storage, read_write> geo2:     array<vec4<f32>>; // com.xyz,    0

@compute @workgroup_size(256)
fn flatten_geometry(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let total = 2u * params.n - 1u;
    if (d >= total) { return; }
    let u = slot_meta[d].w;                                  // unified index at this DFS slot
    let mn = node_min[u].xyz;
    let mx = node_max[u].xyz;
    let cm = node_com[u];
    let center = (mn + mx) * 0.5;
    let half = (mx - mn) * 0.5;
    let com = cm.xyz;
    let delta = length(com - center);
    geo0[d] = vec4<f32>(center, delta);
    geo1[d] = vec4<f32>(half, cm.w);
    geo2[d] = vec4<f32>(com, 0.0);
}
"#;

/// GPU DFS skip-pointer flatten stage. Composes a [`GpuLbvhBuilder`] (M4e pointer-tree
/// build) and drives the two flatten pipelines on its own compute device; storage buffers
/// grow lazily with N. Same bring-up idiom as the other GPU-resident stages.
pub struct GpuLbvhFlattener {
    builder: GpuLbvhBuilder,
    device: wgpu::Device,
    queue: wgpu::Queue,
    structure_pipeline: wgpu::ComputePipeline,
    geometry_pipeline: wgpu::ComputePipeline,
    struct_bgl: wgpu::BindGroupLayout,
    geom_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    // Storage grown lazily to the largest N seen; bind groups rebuilt on (re)alloc.
    children_buf: Option<wgpu::Buffer>,
    parent_buf: Option<wgpu::Buffer>,
    order_buf: Option<wgpu::Buffer>,
    meta_buf: Option<wgpu::Buffer>,
    leaf_bodies_buf: Option<wgpu::Buffer>,
    size_buf: Option<wgpu::Buffer>,
    stack_buf: Option<wgpu::Buffer>,
    node_min_buf: Option<wgpu::Buffer>,
    node_max_buf: Option<wgpu::Buffer>,
    node_com_buf: Option<wgpu::Buffer>,
    geo0_buf: Option<wgpu::Buffer>,
    geo1_buf: Option<wgpu::Buffer>,
    geo2_buf: Option<wgpu::Buffer>,
    meta_rb: Option<wgpu::Buffer>,
    leaf_bodies_rb: Option<wgpu::Buffer>,
    geo0_rb: Option<wgpu::Buffer>,
    geo1_rb: Option<wgpu::Buffer>,
    geo2_rb: Option<wgpu::Buffer>,
    struct_bg: Option<wgpu::BindGroup>,
    geom_bg: Option<wgpu::BindGroup>,
    capacity: usize,
}

impl GpuLbvhFlattener {
    /// Bring up a headless wgpu compute device and the two flatten pipelines (plus the
    /// composed M4e builder). Returns a typed [`GpuError`] (never panics) when no adapter
    /// is available.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let builder = GpuLbvhBuilder::new()?;

        let crate::context::GpuContext { device, queue } = crate::context::gpu_context()?;

        let structure_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-lbvh-flatten-structure"),
            source: wgpu::ShaderSource::Wgsl(STRUCTURE_SHADER.into()),
        });
        let geometry_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-lbvh-flatten-geometry"),
            source: wgpu::ShaderSource::Wgsl(GEOMETRY_SHADER.into()),
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
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

        let struct_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-lbvh-flatten-struct-bgl"),
            entries: &[
                uniform_entry,
                storage(1, true),  // children
                storage(2, true),  // parent
                storage(3, true),  // order
                storage(4, false), // slot_meta (rw)
                storage(5, false), // leaf_bodies (rw)
                storage(6, false), // size (rw scratch)
                storage(7, false), // stack (rw scratch)
            ],
        });
        let geom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-lbvh-flatten-geom-bgl"),
            entries: &[
                uniform_entry,
                storage(1, true),  // slot_meta
                storage(2, true),  // node_min
                storage(3, true),  // node_max
                storage(4, true),  // node_com
                storage(5, false), // geo0 (rw)
                storage(6, false), // geo1 (rw)
                storage(7, false), // geo2 (rw)
            ],
        });

        let struct_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-lbvh-flatten-struct-pl"),
            bind_group_layouts: &[Some(&struct_bgl)],
            immediate_size: 0,
        });
        let geom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-lbvh-flatten-geom-pl"),
            bind_group_layouts: &[Some(&geom_bgl)],
            immediate_size: 0,
        });

        let structure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-lbvh-flatten-structure-pipeline"),
            layout: Some(&struct_pl),
            module: &structure_mod,
            entry_point: Some("flatten_structure"),
            compilation_options: Default::default(),
            cache: None,
        });
        let geometry_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-lbvh-flatten-geometry-pipeline"),
            layout: Some(&geom_pl),
            module: &geometry_mod,
            entry_point: Some("flatten_geometry"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-lbvh-flatten-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuLbvhFlattener {
            builder,
            device,
            queue,
            structure_pipeline,
            geometry_pipeline,
            struct_bgl,
            geom_bgl,
            params_buf,
            children_buf: None,
            parent_buf: None,
            order_buf: None,
            meta_buf: None,
            leaf_bodies_buf: None,
            size_buf: None,
            stack_buf: None,
            node_min_buf: None,
            node_max_buf: None,
            node_com_buf: None,
            geo0_buf: None,
            geo1_buf: None,
            geo2_buf: None,
            meta_rb: None,
            leaf_bodies_rb: None,
            geo0_rb: None,
            geo1_rb: None,
            geo2_rb: None,
            struct_bg: None,
            geom_bg: None,
            capacity: 0,
        })
    }

    /// Ensure the storage / readback buffers hold at least `n` leaves, rebuilding both bind
    /// groups when they are (re)allocated. Only grows. Caller guarantees `n >= 2`.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity && self.struct_bg.is_some() {
            return;
        }
        let total = 2 * n - 1;
        let u32_children = ((n - 1) * 2 * 4) as u64;
        let u32_n = (n * 4) as u64;
        let u32_total = (total * 4) as u64;
        let uvec4_total = (total * 16) as u64;
        let fvec4_total = (total * 16) as u64;

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

        let children_buf = make("gpu-lbvh-flatten-children", u32_children, storage | cdst);
        let parent_buf = make("gpu-lbvh-flatten-parent", u32_total, storage | cdst);
        let order_buf = make("gpu-lbvh-flatten-order", u32_n, storage | cdst);
        let meta_buf = make("gpu-lbvh-flatten-slot_meta", uvec4_total, storage | csrc);
        let leaf_bodies_buf = make("gpu-lbvh-flatten-leaf-bodies", u32_n, storage | csrc);
        let size_buf = make("gpu-lbvh-flatten-size", u32_total, storage);
        let stack_buf = make("gpu-lbvh-flatten-stack", u32_total, storage);
        let node_min_buf = make("gpu-lbvh-flatten-node-min", fvec4_total, storage | cdst);
        let node_max_buf = make("gpu-lbvh-flatten-node-max", fvec4_total, storage | cdst);
        let node_com_buf = make("gpu-lbvh-flatten-node-com", fvec4_total, storage | cdst);
        let geo0_buf = make("gpu-lbvh-flatten-geo0", fvec4_total, storage | csrc);
        let geo1_buf = make("gpu-lbvh-flatten-geo1", fvec4_total, storage | csrc);
        let geo2_buf = make("gpu-lbvh-flatten-geo2", fvec4_total, storage | csrc);

        let meta_rb = make("gpu-lbvh-flatten-slot_meta-rb", uvec4_total, mapread);
        let leaf_bodies_rb = make("gpu-lbvh-flatten-leaf-rb", u32_n, mapread);
        let geo0_rb = make("gpu-lbvh-flatten-geo0-rb", fvec4_total, mapread);
        let geo1_rb = make("gpu-lbvh-flatten-geo1-rb", fvec4_total, mapread);
        let geo2_rb = make("gpu-lbvh-flatten-geo2-rb", fvec4_total, mapread);

        let struct_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-lbvh-flatten-struct-bg"),
            layout: &self.struct_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: children_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: leaf_bodies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: size_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: stack_buf.as_entire_binding(),
                },
            ],
        });
        let geom_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-lbvh-flatten-geom-bg"),
            layout: &self.geom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: node_min_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: node_max_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: node_com_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: geo0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: geo1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: geo2_buf.as_entire_binding(),
                },
            ],
        });

        self.children_buf = Some(children_buf);
        self.parent_buf = Some(parent_buf);
        self.order_buf = Some(order_buf);
        self.meta_buf = Some(meta_buf);
        self.leaf_bodies_buf = Some(leaf_bodies_buf);
        self.size_buf = Some(size_buf);
        self.stack_buf = Some(stack_buf);
        self.node_min_buf = Some(node_min_buf);
        self.node_max_buf = Some(node_max_buf);
        self.node_com_buf = Some(node_com_buf);
        self.geo0_buf = Some(geo0_buf);
        self.geo1_buf = Some(geo1_buf);
        self.geo2_buf = Some(geo2_buf);
        self.meta_rb = Some(meta_rb);
        self.leaf_bodies_rb = Some(leaf_bodies_rb);
        self.geo0_rb = Some(geo0_rb);
        self.geo1_rb = Some(geo1_rb);
        self.geo2_rb = Some(geo2_rb);
        self.struct_bg = Some(struct_bg);
        self.geom_bg = Some(geom_bg);
        self.capacity = n;
    }

    /// Flatten the Karras pointer tree (built on the GPU from `sorted_codes` +
    /// `sorted_pos`/`sorted_mass` via the composed [`GpuLbvhBuilder`]) into the DFS
    /// skip-pointer form. `order[k]` is the original index of the k-th sorted leaf (written
    /// into `leaf_bodies`). `sorted_codes` may be empty (yields an empty flat form). Panics
    /// if the input lengths disagree.
    pub fn build_flat(
        &mut self,
        sorted_codes: &[u32],
        sorted_pos: &[DVec3],
        sorted_mass: &[f64],
        order: &[u32],
    ) -> GpuLbvhFlat {
        let n = sorted_codes.len();
        assert_eq!(sorted_pos.len(), n, "sorted_pos length must equal N");
        assert_eq!(sorted_mass.len(), n, "sorted_mass length must equal N");
        assert_eq!(order.len(), n, "order length must equal N");

        if n == 0 {
            return GpuLbvhFlat::empty();
        }
        if n == 1 {
            // A single leaf is the whole flat tree — no dispatch (matches the reference).
            let p = [
                sorted_pos[0].x as f32,
                sorted_pos[0].y as f32,
                sorted_pos[0].z as f32,
            ];
            return GpuLbvhFlat {
                n: 1,
                center: vec![p],
                half_extents: vec![[0.0, 0.0, 0.0]],
                com: vec![p],
                mass: vec![sorted_mass[0] as f32],
                delta: vec![0.0],
                next: vec![1],
                body_start: vec![0],
                body_count: vec![1],
                leaf_bodies: vec![order[0]],
            };
        }

        // Build the M4e pointer tree on the GPU, then re-upload it for the flatten.
        let tree = self.builder.build(sorted_codes, sorted_pos, sorted_mass);
        let total = 2 * n - 1;

        // Pack the pointer tree for the flatten device: children interleaved (L at 2i, R at
        // 2i+1), and the aggregate AABB/com/mass as vec4 (com carries mass in w).
        let children: Vec<u32> = (0..n - 1)
            .flat_map(|i| [tree.left[i], tree.right[i]])
            .collect();
        let node_min: Vec<[f32; 4]> = tree
            .aabb_min
            .iter()
            .map(|c| [c[0], c[1], c[2], 0.0])
            .collect();
        let node_max: Vec<[f32; 4]> = tree
            .aabb_max
            .iter()
            .map(|c| [c[0], c[1], c[2], 0.0])
            .collect();
        let node_com: Vec<[f32; 4]> = tree
            .com
            .iter()
            .zip(&tree.mass)
            .map(|(c, &m)| [c[0], c[1], c[2], m])
            .collect();

        self.ensure_capacity(n);

        let params = Params {
            n: n as u32,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let write = |buf: &Option<wgpu::Buffer>, data: &[u8]| {
            self.queue
                .write_buffer(buf.as_ref().expect("buffers ensured above"), 0, data);
        };
        write(&self.children_buf, bytemuck::cast_slice(&children));
        write(&self.parent_buf, bytemuck::cast_slice(&tree.parent));
        write(&self.order_buf, bytemuck::cast_slice(order));
        write(&self.node_min_buf, bytemuck::cast_slice(&node_min));
        write(&self.node_max_buf, bytemuck::cast_slice(&node_max));
        write(&self.node_com_buf, bytemuck::cast_slice(&node_com));

        let struct_bg = self.struct_bg.as_ref().expect("bind group ensured above");
        let geom_bg = self.geom_bg.as_ref().expect("bind group ensured above");

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Single-invocation DFS structure pass, then the parallel geometry gather.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-lbvh-flatten-structure-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.structure_pipeline);
            pass.set_bind_group(0, struct_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single invocation (see module doc)
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-lbvh-flatten-geometry-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.geometry_pipeline);
            pass.set_bind_group(0, geom_bg, &[]);
            pass.dispatch_workgroups((total as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        // Copy results to the mappable readback buffers.
        let uvec4_total = (total * 16) as u64;
        let u32_n = (n * 4) as u64;
        let meta_buf = self.meta_buf.as_ref().unwrap();
        let leaf_bodies_buf = self.leaf_bodies_buf.as_ref().unwrap();
        let geo0_buf = self.geo0_buf.as_ref().unwrap();
        let geo1_buf = self.geo1_buf.as_ref().unwrap();
        let geo2_buf = self.geo2_buf.as_ref().unwrap();
        let meta_rb = self.meta_rb.as_ref().unwrap();
        let leaf_bodies_rb = self.leaf_bodies_rb.as_ref().unwrap();
        let geo0_rb = self.geo0_rb.as_ref().unwrap();
        let geo1_rb = self.geo1_rb.as_ref().unwrap();
        let geo2_rb = self.geo2_rb.as_ref().unwrap();
        enc.copy_buffer_to_buffer(meta_buf, 0, meta_rb, 0, uvec4_total);
        enc.copy_buffer_to_buffer(leaf_bodies_buf, 0, leaf_bodies_rb, 0, u32_n);
        enc.copy_buffer_to_buffer(geo0_buf, 0, geo0_rb, 0, uvec4_total);
        enc.copy_buffer_to_buffer(geo1_buf, 0, geo1_rb, 0, uvec4_total);
        enc.copy_buffer_to_buffer(geo2_buf, 0, geo2_rb, 0, uvec4_total);
        self.queue.submit([enc.finish()]);

        // Map all five readbacks, wait once, read. A map failure is a genuine GPU loss.
        let slices = [
            meta_rb.slice(..uvec4_total),
            leaf_bodies_rb.slice(..u32_n),
            geo0_rb.slice(..uvec4_total),
            geo1_rb.slice(..uvec4_total),
            geo2_rb.slice(..uvec4_total),
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

        // slot_meta = [next, body_start, body_count, unified] per DFS slot.
        let meta_data = slices[0].get_mapped_range();
        let slot_meta: &[u32] = bytemuck::cast_slice(&meta_data);
        let next: Vec<u32> = (0..total).map(|d| slot_meta[d * 4]).collect();
        let body_start: Vec<u32> = (0..total).map(|d| slot_meta[d * 4 + 1]).collect();
        let body_count: Vec<u32> = (0..total).map(|d| slot_meta[d * 4 + 2]).collect();
        drop(meta_data);

        let leaf_bodies: Vec<u32> =
            bytemuck::cast_slice::<u8, u32>(&slices[1].get_mapped_range())[..n].to_vec();

        let unpack3 = |slice: &wgpu::BufferSlice| -> Vec<[f32; 3]> {
            let data = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&data);
            (0..total)
                .map(|d| [f[d * 4], f[d * 4 + 1], f[d * 4 + 2]])
                .collect()
        };
        let geo0_data = slices[2].get_mapped_range();
        let g0: &[f32] = bytemuck::cast_slice(&geo0_data);
        let center: Vec<[f32; 3]> = (0..total)
            .map(|d| [g0[d * 4], g0[d * 4 + 1], g0[d * 4 + 2]])
            .collect();
        let delta: Vec<f32> = (0..total).map(|d| g0[d * 4 + 3]).collect();
        drop(geo0_data);
        let geo1_data = slices[3].get_mapped_range();
        let g1: &[f32] = bytemuck::cast_slice(&geo1_data);
        let half_extents: Vec<[f32; 3]> = (0..total)
            .map(|d| [g1[d * 4], g1[d * 4 + 1], g1[d * 4 + 2]])
            .collect();
        let mass: Vec<f32> = (0..total).map(|d| g1[d * 4 + 3]).collect();
        drop(geo1_data);
        let com = unpack3(&slices[4]);

        meta_rb.unmap();
        leaf_bodies_rb.unmap();
        geo0_rb.unmap();
        geo1_rb.unmap();
        geo2_rb.unmap();

        GpuLbvhFlat {
            n,
            center,
            half_extents,
            com,
            mass,
            delta,
            next,
            body_start,
            body_count,
            leaf_bodies,
        }
    }
}
