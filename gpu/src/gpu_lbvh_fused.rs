//! [`GpuLbvhFused`]: the single-device, GPU-resident **fuse** of the M4c–M4g LBVH pipeline
//! (DESIGN M4h) — the named scale refinement [`crate::GpuLbvh`] (M4g) deferred.
//!
//! [`crate::GpuLbvh`] (M4g) is the *reference-grade composition*: each build stage owns its own
//! wgpu device and the pointer tree / flat form round-trips through host memory between stages,
//! so a `GpuLbvh` holds several devices and pays ~5 CPU↔GPU sync points (readback + reupload)
//! per `accelerations` call. `GpuLbvhFused` runs the **whole pipeline on one device in one
//! submit**: `bodies` are uploaded once, every intermediate — Morton codes → sorted order →
//! gathered leaves → Karras pointer tree → DFS skip-pointer flat form — stays in GPU storage
//! buffers that flow directly from one compute pass to the next (wgpu's automatic usage-tracked
//! barriers order the passes), and only the final `accel` is read back. One upload + one
//! readback — replacing the reference chain's ~5 readback/reupload round-trips (one per stage:
//! morton, sort, tree-build, flatten, traverse) with a single submit (≈4 fewer sync points).
//!
//! ## Same forces, same interface — a lossless refactor
//! Every stage runs the **same f32 WGSL** as the M4g chain: the module-level `SHADER` consts of
//! [`crate::lbvh_morton`], [`crate::lbvh_sort`], [`crate::lbvh_tree`],
//! [`crate::lbvh_flatten`] (structure), and [`crate::gpu_lbvh`] (the complex traversal) are
//! reused verbatim. Only two *trivial* kernels are new: a `gather` that pulls each sorted
//! leaf's `(pos, mass)` from the uploaded `bodies` (the host did this between stages in the
//! reference), and a geometry kernel that derives `center`/`half`/`delta`/`com`/`mass` and
//! writes them in the **traversal's** buffer packing (so the reference host-side repack
//! vanishes). Because the values are computed by identical arithmetic, on a given device this
//! reproduces the reference `GpuLbvh` forces — see the M4h faithful-refactor gate. `(g,
//! softening, theta)` semantics and the `ForceSolver` interface are unchanged.
//!
//! ## Scope: this fuses the *build pipeline*, not cross-step residency
//! M4h keeps particle state on the GPU across the **stages of one force evaluation**. Keeping
//! state GPU-resident across **integrator steps** (which would change the
//! `accelerations(&State)→acc` interface and touch the stepping loop) is a *separate* deferred
//! item — see DESIGN "Remaining M4+". This is a latency / architecture win (one submit; ≈4
//! fewer CPU↔GPU sync points), the precondition for that residency, **not** a throughput speedup: the
//! single-invocation serial stages (sort, aggregate, flatten-structure) are unchanged and stay
//! the bottleneck; their parallel refinements remain deferred.

use bytemuck::{Pod, Zeroable};

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::NO_PARENT;

use crate::lbvh_sort::{NUM_PASSES, RADIX_BITS};
use crate::GpuError;

/// Workgroup width for the wide (per-element) passes — matches the reused kernels
/// (`quantize`, `build_tree`, geometry all declare `@workgroup_size(256)`).
const WG_WIDE: u32 = 256;
/// Workgroup width for the traversal (matches the reused `gpu_lbvh` kernel's `@workgroup_size(64)`).
const WG_TRAVERSE: u32 = 64;

/// New trivial kernel: gather each sorted leaf's `(pos, mass)` from the uploaded `bodies` via
/// the sort permutation `order` (= the sort's final `idx` buffer). This is the on-device
/// equivalent of the reference chain's host-side `order.iter().map(|&i| state.pos[i])` gather;
/// because `bodies` already holds the f32-narrowed positions the reference builder would have
/// received, the gathered payload is bit-identical to the reference's.
const GATHER_SHADER: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read>       bodies:      array<vec4<f32>>; // xyz=pos, w=mass
@group(0) @binding(2) var<storage, read>       order:       array<u32>;
@group(0) @binding(3) var<storage, read_write> sorted_leaf: array<vec4<f32>>;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= params.n) { return; }
    sorted_leaf[k] = bodies[order[k]];
}
"#;

/// New trivial kernel: the DFS-slot geometry gather. Same math as the reference
/// `lbvh_flatten::GEOMETRY_SHADER` (`center=(min+max)/2`, `half=(max-min)/2`,
/// `delta=|com-center|`) but written straight into the **traversal's** three buffers —
/// `node_center=[center, delta]`, `node_half=[half, 0]`, `node_cm=[com, mass]` — so no host
/// repack sits between flatten and traverse. `slot_meta[d].w` is the DFS slot's unified index
/// (the structure pass wrote it there), used to gather the M4e aggregate.
const GEOMETRY_SHADER: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read>       slot_meta:   array<vec4<u32>>; // per DFS slot
@group(0) @binding(2) var<storage, read>       node_min:    array<vec4<f32>>; // per unified node
@group(0) @binding(3) var<storage, read>       node_max:    array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       node_com:    array<vec4<f32>>; // xyz=com, w=mass
@group(0) @binding(5) var<storage, read_write> node_center: array<vec4<f32>>; // xyz=center, w=delta
@group(0) @binding(6) var<storage, read_write> node_half:   array<vec4<f32>>; // xyz=half,   w=0
@group(0) @binding(7) var<storage, read_write> node_cm:     array<vec4<f32>>; // xyz=com,    w=mass

@compute @workgroup_size(256)
fn flatten_geometry(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let total = 2u * params.n - 1u;
    if (d >= total) { return; }
    let u = slot_meta[d].w;
    let mn = node_min[u].xyz;
    let mx = node_max[u].xyz;
    let cm = node_com[u];
    let center = (mn + mx) * 0.5;
    let half = (mx - mn) * 0.5;
    let com = cm.xyz;
    let delta = length(com - center);
    node_center[d] = vec4<f32>(center, delta);
    node_half[d]   = vec4<f32>(half, 0.0);
    node_cm[d]     = vec4<f32>(com, cm.w);
}
"#;

/// Uniform block carrying just `n` — mirrors the WGSL `Params { n, pad×3 }` shared by the
/// morton, gather, tree, flatten-structure and geometry kernels (they all read only `.n`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NParams {
    n: u32,
    _pad: [u32; 3],
}

/// Uniform block for one radix pass: `n` + the pass's constant shift (`0/8/16/24`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortParams {
    n: u32,
    shift: u32,
    _pad: [u32; 2],
}

/// Uniform block for the traversal — mirrors [`crate::gpu_lbvh`]'s `Params` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraverseParams {
    n: u32,
    n_nodes: u32,
    eps2: f32,
    g: f32,
    theta: f32,
    _pad: [u32; 3],
}

/// The per-capacity resource set: the host-touched buffers + the per-stage bind groups,
/// (re)allocated together as `N` grows.
///
/// The many *intermediate* buffers (Morton codes, the ping-pong sort keys, the gathered leaves,
/// the Karras pointer tree, the DFS flat form, the flatten scratch) are deliberately **not**
/// stored here: each is bound once into the bind groups below, and a wgpu bind group retains its
/// resources for its own lifetime, so those buffers live exactly as long as the bind groups that
/// use them. Only the buffers the host re-touches every force evaluation are kept —
/// `bodies`/`idx_a`/`parent` (uploaded), `counter` (cleared), `accel`+`readback` (the single
/// readback). Growing rebuilds this whole struct, dropping the old bind groups (and with them
/// the old intermediates), so there is no leak.
struct FusedResources {
    bodies: wgpu::Buffer,
    idx_a: wgpu::Buffer,
    parent: wgpu::Buffer,
    counter: wgpu::Buffer,
    accel: wgpu::Buffer,
    readback: wgpu::Buffer,
    morton_bg: wgpu::BindGroup,
    sort_bgs: [wgpu::BindGroup; NUM_PASSES as usize],
    gather_bg: wgpu::BindGroup,
    tree_bg: wgpu::BindGroup,
    struct_bg: wgpu::BindGroup,
    geom_bg: wgpu::BindGroup,
    traverse_bg: wgpu::BindGroup,
    capacity: usize,
}

/// GPU Barnes-Hut force solver over a GPU-resident Morton Linear BVH, **fused onto a single
/// wgpu device** — the M4h refinement of [`crate::GpuLbvh`]. Same `(g, softening, theta)`
/// semantics; one upload + one readback per force evaluation.
pub struct GpuLbvhFused {
    g: f64,
    softening: f64,
    theta: f64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Pipelines (built once). reduce/quantize = morton; radix = sort; build/aggregate = tree.
    reduce_pl: wgpu::ComputePipeline,
    quantize_pl: wgpu::ComputePipeline,
    radix_pl: wgpu::ComputePipeline,
    gather_pl: wgpu::ComputePipeline,
    build_pl: wgpu::ComputePipeline,
    aggregate_pl: wgpu::ComputePipeline,
    structure_pl: wgpu::ComputePipeline,
    geometry_pl: wgpu::ComputePipeline,
    traverse_pl: wgpu::ComputePipeline,
    // Bind-group layouts (built once).
    morton_bgl: wgpu::BindGroupLayout,
    sort_bgl: wgpu::BindGroupLayout,
    gather_bgl: wgpu::BindGroupLayout,
    tree_bgl: wgpu::BindGroupLayout,
    struct_bgl: wgpu::BindGroupLayout,
    geom_bgl: wgpu::BindGroupLayout,
    traverse_bgl: wgpu::BindGroupLayout,
    // Fixed-size uniform buffers (built once).
    n_params_buf: wgpu::Buffer,
    sort_params_bufs: [wgpu::Buffer; NUM_PASSES as usize],
    traverse_params_buf: wgpu::Buffer,
    // Lazily-sized storage + bind groups.
    res: Option<FusedResources>,
}

/// A read-only storage binding entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A bind-group entry binding a whole buffer (a free `fn`, not a closure, so the returned
/// entry's lifetime can be tied to `buf`).
fn bg_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

/// A uniform binding entry.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuLbvhFused {
    /// Bring up the single fused compute device + every pipeline. Returns a typed [`GpuError`]
    /// (never panics) when no adapter is available.
    ///
    /// Requests **no** device features (baseline storage-buffer compute); the widest bind group
    /// (the tree build/aggregate) uses 8 storage buffers + 1 uniform, within the default
    /// `maxStorageBuffersPerShaderStage` of 8 — so, like every other GPU stage, it does not
    /// narrow adapter support.
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
                label: Some("galaxy-gpu-lbvh-fused-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        // Compile the reused (verbatim) kernels + the two new trivial ones as separate modules.
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let morton_mod = module("fused-morton", crate::lbvh_morton::SHADER);
        let sort_mod = module("fused-sort", crate::lbvh_sort::SHADER);
        let gather_mod = module("fused-gather", GATHER_SHADER);
        let tree_mod = module("fused-tree", crate::lbvh_tree::SHADER);
        let struct_mod = module(
            "fused-flatten-structure",
            crate::lbvh_flatten::STRUCTURE_SHADER,
        );
        let geom_mod = module("fused-flatten-geometry", GEOMETRY_SHADER);
        let traverse_mod = module("fused-traverse", crate::gpu_lbvh::SHADER);

        // --- Bind-group layouts (one per distinct binding scheme; must match each WGSL). ---
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        // morton: 0 positions(r), 1 bbox(rw), 2 lanes(rw), 3 codes(rw), 4 uniform
        let morton_bgl = bgl(
            "fused-morton-bgl",
            &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        );
        // sort: 0 uniform, 1 src_keys(r), 2 src_idx(r), 3 dst_keys(rw), 4 dst_idx(rw)
        let sort_bgl = bgl(
            "fused-sort-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        );
        // gather: 0 uniform, 1 bodies(r), 2 order(r), 3 sorted_leaf(rw)
        let gather_bgl = bgl(
            "fused-gather-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        );
        // tree: 0 uniform, 1 codes(r), 2 leaf(r), 3 children(rw), 4 parent(rw), 5 node_min(rw),
        // 6 node_max(rw), 7 node_com(rw), 8 counter(rw)
        let tree_bgl = bgl(
            "fused-tree-bgl",
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
        // flatten-structure: 0 uniform, 1 children(r), 2 parent(r), 3 order(r), 4 slot_meta(rw),
        // 5 leaf_bodies(rw), 6 size(rw), 7 stack(rw)
        let struct_bgl = bgl(
            "fused-struct-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        );
        // geometry: 0 uniform, 1 slot_meta(r), 2 node_min(r), 3 node_max(r), 4 node_com(r),
        // 5 node_center(rw), 6 node_half(rw), 7 node_cm(rw)
        let geom_bgl = bgl(
            "fused-geom-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        );
        // traverse: 0 bodies(r), 1 node_center(r), 2 node_half(r), 3 node_cm(r), 4 node_meta(r),
        // 5 leaf_bodies(r), 6 accel(rw), 7 uniform
        let traverse_bgl = bgl(
            "fused-traverse-bgl",
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                uniform_entry(7),
            ],
        );

        // --- Pipelines. ---
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
        let reduce_pl = pipeline("fused-reduce", &morton_bgl, &morton_mod, "reduce");
        let quantize_pl = pipeline("fused-quantize", &morton_bgl, &morton_mod, "quantize");
        let radix_pl = pipeline("fused-radix", &sort_bgl, &sort_mod, "radix_pass");
        let gather_pl = pipeline("fused-gather", &gather_bgl, &gather_mod, "gather");
        let build_pl = pipeline("fused-build", &tree_bgl, &tree_mod, "build_tree");
        let aggregate_pl = pipeline("fused-aggregate", &tree_bgl, &tree_mod, "aggregate");
        let structure_pl = pipeline(
            "fused-structure",
            &struct_bgl,
            &struct_mod,
            "flatten_structure",
        );
        let geometry_pl = pipeline("fused-geometry", &geom_bgl, &geom_mod, "flatten_geometry");
        let traverse_pl = pipeline("fused-traverse", &traverse_bgl, &traverse_mod, "main");

        // --- Fixed-size uniform buffers. ---
        let uniform_buf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let n_params_buf = uniform_buf("fused-n-params", std::mem::size_of::<NParams>() as u64);
        let sort_params_bufs = std::array::from_fn(|_| {
            uniform_buf(
                "fused-sort-params",
                std::mem::size_of::<SortParams>() as u64,
            )
        });
        let traverse_params_buf = uniform_buf(
            "fused-traverse-params",
            std::mem::size_of::<TraverseParams>() as u64,
        );

        Ok(GpuLbvhFused {
            g,
            softening,
            theta,
            device,
            queue,
            reduce_pl,
            quantize_pl,
            radix_pl,
            gather_pl,
            build_pl,
            aggregate_pl,
            structure_pl,
            geometry_pl,
            traverse_pl,
            morton_bgl,
            sort_bgl,
            gather_bgl,
            tree_bgl,
            struct_bgl,
            geom_bgl,
            traverse_bgl,
            n_params_buf,
            sort_params_bufs,
            traverse_params_buf,
            res: None,
        })
    }

    /// (Re)allocate every lazily-sized storage buffer + bind group to hold `n` bodies (and the
    /// derived `2N-1` nodes). Only grows: a later smaller problem reuses the larger buffers
    /// (kernels bound themselves by the uniform `n`). Caller guarantees `n >= 2`.
    fn ensure_capacity(&mut self, n: usize) {
        if let Some(res) = &self.res {
            if n <= res.capacity {
                return;
            }
        }
        let total = 2 * n - 1;
        let dev = &self.device;
        let store = wgpu::BufferUsages::STORAGE;
        let cdst = wgpu::BufferUsages::COPY_DST;
        let csrc = wgpu::BufferUsages::COPY_SRC;
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let f4 = |count: usize| (count * std::mem::size_of::<[f32; 4]>()) as u64;
        let u4 = |count: usize| (count * std::mem::size_of::<[u32; 4]>()) as u64;
        let u1 = |count: usize| (count * std::mem::size_of::<u32>()) as u64;

        // bodies: uploaded, read by morton/gather/traverse. lanes: morton scratch (unused
        // downstream but the reused quantize kernel writes it). key_a/idx_a seeded (idx via
        // upload); result of the 4-pass sort lands back in A. parent: NO_PARENT-initialised
        // upload, then build_tree overwrites non-root slots. counter: cleared each call.
        let bodies = make("fused-bodies", f4(n), store | cdst);
        let bbox = make("fused-bbox", 32, store);
        let lanes = make("fused-lanes", u4(n), store);
        let key_a = make("fused-key-a", u1(n), store);
        let key_b = make("fused-key-b", u1(n), store);
        let idx_a = make("fused-idx-a", u1(n), store | cdst);
        let idx_b = make("fused-idx-b", u1(n), store);
        let sorted_leaf = make("fused-sorted-leaf", f4(n), store);
        let leaf_bodies = make("fused-leaf-bodies", u1(n), store);
        let accel = make("fused-accel", f4(n), store | csrc);
        let readback = make("fused-readback", f4(n), cdst | wgpu::BufferUsages::MAP_READ);

        let children = make("fused-children", u1(2 * (n - 1)), store);
        let counter = make("fused-counter", u1(n - 1), store | cdst);

        let parent = make("fused-parent", u1(total), store | cdst);
        let node_min = make("fused-node-min", f4(total), store);
        let node_max = make("fused-node-max", f4(total), store);
        let node_com = make("fused-node-com", f4(total), store);
        let slot_meta = make("fused-slot-meta", u4(total), store);
        let size = make("fused-size", u1(total), store);
        let stack = make("fused-stack", u1(total), store);
        let node_center = make("fused-node-center", f4(total), store);
        let node_half = make("fused-node-half", f4(total), store);
        let node_cm = make("fused-node-cm", f4(total), store);

        // --- Bind groups (buffer → binding, matching each layout above). ---
        let entry = bg_entry;
        let bind =
            |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
                dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries,
                })
            };

        let morton_bg = bind(
            "fused-morton-bg",
            &self.morton_bgl,
            &[
                entry(0, &bodies),
                entry(1, &bbox),
                entry(2, &lanes),
                entry(3, &key_a),
                entry(4, &self.n_params_buf),
            ],
        );
        // Pass p reads src, writes dst: even A→B, odd B→A. After NUM_PASSES (even) the result
        // (sorted codes + order) is back in key_a/idx_a.
        let sort_bgs: [wgpu::BindGroup; NUM_PASSES as usize] = std::array::from_fn(|p| {
            let (src_k, src_i, dst_k, dst_i) = if p % 2 == 0 {
                (&key_a, &idx_a, &key_b, &idx_b)
            } else {
                (&key_b, &idx_b, &key_a, &idx_a)
            };
            bind(
                "fused-sort-bg",
                &self.sort_bgl,
                &[
                    entry(0, &self.sort_params_bufs[p]),
                    entry(1, src_k),
                    entry(2, src_i),
                    entry(3, dst_k),
                    entry(4, dst_i),
                ],
            )
        });
        let gather_bg = bind(
            "fused-gather-bg",
            &self.gather_bgl,
            &[
                entry(0, &self.n_params_buf),
                entry(1, &bodies),
                entry(2, &idx_a),
                entry(3, &sorted_leaf),
            ],
        );
        let tree_bg = bind(
            "fused-tree-bg",
            &self.tree_bgl,
            &[
                entry(0, &self.n_params_buf),
                entry(1, &key_a), // sorted codes after the sort
                entry(2, &sorted_leaf),
                entry(3, &children),
                entry(4, &parent),
                entry(5, &node_min),
                entry(6, &node_max),
                entry(7, &node_com),
                entry(8, &counter),
            ],
        );
        let struct_bg = bind(
            "fused-struct-bg",
            &self.struct_bgl,
            &[
                entry(0, &self.n_params_buf),
                entry(1, &children),
                entry(2, &parent),
                entry(3, &idx_a), // order
                entry(4, &slot_meta),
                entry(5, &leaf_bodies),
                entry(6, &size),
                entry(7, &stack),
            ],
        );
        let geom_bg = bind(
            "fused-geom-bg",
            &self.geom_bgl,
            &[
                entry(0, &self.n_params_buf),
                entry(1, &slot_meta),
                entry(2, &node_min),
                entry(3, &node_max),
                entry(4, &node_com),
                entry(5, &node_center),
                entry(6, &node_half),
                entry(7, &node_cm),
            ],
        );
        let traverse_bg = bind(
            "fused-traverse-bg",
            &self.traverse_bgl,
            &[
                entry(0, &bodies),
                entry(1, &node_center),
                entry(2, &node_half),
                entry(3, &node_cm),
                entry(4, &slot_meta), // node_meta (w=unified ignored by the traversal)
                entry(5, &leaf_bodies),
                entry(6, &accel),
                entry(7, &self.traverse_params_buf),
            ],
        );

        // Store only the host-touched buffers + the bind groups. The intermediates (bbox, lanes,
        // key_a/b, idx_b, sorted_leaf, leaf_bodies, children, node_min/max/com, slot_meta, size,
        // stack, node_center/half/cm) are dropped here but kept alive by the bind groups above.
        self.res = Some(FusedResources {
            bodies,
            idx_a,
            parent,
            counter,
            accel,
            readback,
            morton_bg,
            sort_bgs,
            gather_bg,
            tree_bg,
            struct_bg,
            geom_bg,
            traverse_bg,
            capacity: n,
        });
    }
}

impl ForceSolver for GpuLbvhFused {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        if n == 1 {
            // A lone particle feels no force (its only leaf holds just itself) — no dispatch.
            acc[0] = DVec3::ZERO;
            return;
        }

        let total = 2 * n - 1;
        self.ensure_capacity(n);
        let res = self.res.as_ref().expect("resources ensured above");

        // --- The only host→device uploads: bodies, the sort's index seed, and the parent
        // NO_PARENT init. `write_buffer` copies are scheduled before the submitted commands, so
        // they land before any compute pass. The f64→f32 narrowing here is the crate's owned
        // precision reduction; `bodies` (xyz=pos, w=mass) feeds morton, gather AND the traversal.
        let bodies: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let p = state.pos[i];
                [p.x as f32, p.y as f32, p.z as f32, state.mass[i] as f32]
            })
            .collect();
        self.queue
            .write_buffer(&res.bodies, 0, bytemuck::cast_slice(&bodies));
        let idx0: Vec<u32> = (0..n as u32).collect();
        self.queue
            .write_buffer(&res.idx_a, 0, bytemuck::cast_slice(&idx0));
        let parent_init = vec![NO_PARENT; total];
        self.queue
            .write_buffer(&res.parent, 0, bytemuck::cast_slice(&parent_init));

        // Uniforms.
        self.queue.write_buffer(
            &self.n_params_buf,
            0,
            bytemuck::bytes_of(&NParams {
                n: n as u32,
                _pad: [0; 3],
            }),
        );
        for p in 0..NUM_PASSES as usize {
            self.queue.write_buffer(
                &self.sort_params_bufs[p],
                0,
                bytemuck::bytes_of(&SortParams {
                    n: n as u32,
                    shift: p as u32 * RADIX_BITS,
                    _pad: [0; 2],
                }),
            );
        }
        self.queue.write_buffer(
            &self.traverse_params_buf,
            0,
            bytemuck::bytes_of(&TraverseParams {
                n: n as u32,
                n_nodes: total as u32,
                eps2: (self.softening * self.softening) as f32,
                g: self.g as f32,
                theta: self.theta as f32,
                _pad: [0; 3],
            }),
        );

        // --- One command encoder: the whole build + traverse. Each stage is its own compute
        // pass, so wgpu's usage tracking inserts the read-after-write barriers between them
        // (the same cross-pass dependency the M4c/M4e/M4f stages already rely on). ---
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused-lbvh-encoder"),
            });
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

        let wide = |count: usize| (count as u32).div_ceil(WG_WIDE);
        // Morton: single-workgroup bbox reduction, then per-particle quantize (→ codes in key_a).
        pass(&mut enc, "fused-reduce", &self.reduce_pl, &res.morton_bg, 1);
        pass(
            &mut enc,
            "fused-quantize",
            &self.quantize_pl,
            &res.morton_bg,
            wide(n),
        );
        // Sort: NUM_PASSES single-invocation radix passes (ping-pong; result back in key_a/idx_a).
        for p in 0..NUM_PASSES as usize {
            pass(&mut enc, "fused-radix", &self.radix_pl, &res.sort_bgs[p], 1);
        }
        // Gather sorted leaves, build the Karras tree, aggregate (after zeroing the visit counter).
        pass(
            &mut enc,
            "fused-gather",
            &self.gather_pl,
            &res.gather_bg,
            wide(n),
        );
        pass(
            &mut enc,
            "fused-build",
            &self.build_pl,
            &res.tree_bg,
            (n as u32 - 1).div_ceil(WG_WIDE),
        );
        enc.clear_buffer(&res.counter, 0, None);
        pass(
            &mut enc,
            "fused-aggregate",
            &self.aggregate_pl,
            &res.tree_bg,
            1,
        );
        // Flatten: single-invocation DFS structure, then parallel geometry (→ traversal packing).
        pass(
            &mut enc,
            "fused-structure",
            &self.structure_pl,
            &res.struct_bg,
            1,
        );
        pass(
            &mut enc,
            "fused-geometry",
            &self.geometry_pl,
            &res.geom_bg,
            wide(total),
        );
        // Traverse: one invocation per target writes accel[i] once.
        pass(
            &mut enc,
            "fused-traverse",
            &self.traverse_pl,
            &res.traverse_bg,
            (n as u32).div_ceil(WG_TRAVERSE),
        );

        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        enc.copy_buffer_to_buffer(&res.accel, 0, &res.readback, 0, bytes);
        self.queue.submit([enc.finish()]);

        // The single readback: map, block once, widen f32 accelerations back to f64. A map
        // failure is an exceptional GPU loss and panics rather than corrupt state.
        let slice = res.readback.slice(..bytes);
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
        res.readback.unmap();
    }

    /// Softened potential energy, delegated to the shared CPU **f64** reduction — identical to
    /// `GpuLbvh`/`GpuTree`/`BarnesHut`. Same documented inconsistency: forces are f32 while this
    /// is f64, so an energy-drift diagnostic mixes a precision gap with integrator error; it is
    /// a periodic O(N²) diagnostic, not the per-step path.
    fn potential_energy(&self, state: &State) -> f64 {
        galaxy_solvers::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}
