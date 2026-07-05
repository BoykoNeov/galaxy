//! [`GpuNeighborGrid`]: GPU fixed-radius neighbor search for SPH (GPU-SPH G1).
//!
//! A Green-style **counting-sort spatial hash** over gas positions: cell coords
//! `floor(p/cell)` are hashed into a fixed-size table (NOT a dense array — a
//! dense linear-index grid explodes on far merger debris; NOT the CPU's sparse
//! `HashMap` either), particles are counting-sorted into their buckets, and a
//! query GATHERS per target over the `ceil(r/cell)`-cell neighborhood, testing
//! true distance at each candidate. It is the GPU analogue of the CPU
//! [`galaxy_solvers::sph::HashGrid`] and the first stage of the GPU-SPH port
//! (density → hydro force → CFL build on top of it).
//!
//! ## Grid-first, endpoint is the LBVH range query (D4)
//! The measured gas smoothing-length range (34×+ even in the undisturbed disk,
//! ~280× at pericenter) puts this firmly in the regime where a single-resolution
//! grid degenerates *at scale*, so the scale-forward endpoint is a max-h-augmented
//! LBVH range query reusing the Karras construction. This grid is the **grid-first
//! de-risk**: it brings up density/force/CFL against a known-correct, simple
//! neighbor structure before the novel, conservativeness-sensitive LBVH traversal
//! is also in the mix. It is kept isolated behind [`GpuNeighborGrid::query_all`] so
//! the grid↔LBVH swap is a module change, and it survives as a CPU-parity oracle /
//! small-N fallback afterward — not throwaway. See `kindled-resident-cascade.md`.
//!
//! ## Gate: the FILTERED pair set (swap-stable), not raw candidates
//! Correctness is gated as equality of the **filtered pair set** — pairs `(i,j)`
//! with `r_ij < SUPPORT·max(h_i,h_j)` (the true averaged-kernel coupling range) —
//! against `HashGrid`, NOT the raw candidate set. The raw candidate radius is a
//! *policy* (fork(a) global `SUPPORT·h_max` over-gather here; fork(b)/LBVH's
//! per-particle `SUPPORT·h_i` + prune later) that differs between structures while
//! the filtered set is invariant; gating the filtered set is what lets the LBVH
//! swap in without a false gate failure.
//!
//! ## Structure: single-invocation build, per-target-parallel query (D4 walk cap)
//! The **build** (histogram → exclusive scan → stable bucket scatter, the counting
//! sort) is a single `@workgroup_size(1)` invocation — O(N), and the serial scatter
//! is the one place parallelism would reintroduce nondeterminism (same discipline as
//! [`crate::GpuSorter`]). The **query** is per-target parallel (one invocation per
//! particle), two-pass: a count pass fills `nbr_count[i]`, the host exclusive-scans
//! it to CSR row offsets, then a fill pass writes each target's slice. A query is a
//! *gather* (each thread owns its own count and its own output slice — no scatter
//! race), so parallel here is the simple correct thing, still deterministic run-to-run
//! on a given device.
//!
//! The query bucket edge is **capped** at `cell_eff = max(cell, radius/K)` (K = 4),
//! so the neighborhood walk is always ≤ `K` cells per axis (≤ 9³ cells). This is
//! correctness-neutral — a coarser bucket only means larger buckets (more candidates
//! examined, all filtered by true distance), never a missed neighbor — and it is what
//! makes the wide-`h` regime (`radius ≫ cell`) feasible on a uniform grid at all.
//! That a uniform grid *cannot* efficiently serve `cell ≪ radius` without this cap is
//! precisely **D4**'s finding, and precisely why the scale endpoint is the LBVH.

use bytemuck::{Pod, Zeroable};

use galaxy_core::DVec3;

use crate::GpuError;

/// Walk cap: the query bucket edge is at least `radius / WALK_CAP`, so the
/// neighborhood walk spans at most `WALK_CAP` cells per axis. Correctness-neutral
/// (a coarser bucket only enlarges buckets); bounds the `radius ≫ cell` walk (D4).
const WALK_CAP: f64 = 4.0;

/// Per-target neighbor candidate lists in CSR form: `flat[starts[i]..starts[i+1]]`
/// are the indices `j` (INCLUDING `j == i`, matching `HashGrid::neighbours_within`)
/// with `|pos[j] − pos[i]| ≤ radius`. Set-valued: consumers that need the exact
/// `HashGrid` ascending order should sort a copy (the G1 gate compares sets).
pub struct GpuNeighbours {
    /// CSR row offsets, length `n + 1`.
    starts: Vec<u32>,
    /// Concatenated candidate indices, length `starts[n]`.
    flat: Vec<u32>,
}

impl GpuNeighbours {
    /// Candidate indices for query point `i` (includes `i` itself).
    pub fn neighbours(&self, i: usize) -> &[u32] {
        let s = self.starts[i] as usize;
        let e = self.starts[i + 1] as usize;
        &self.flat[s..e]
    }

    /// Number of query points.
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Whether there are no query points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The build (counting sort) + query (count/fill) kernels. `build` is a single
/// invocation; `query_count`/`query_fill` are per-target parallel and call the same
/// `gather` traversal so the fill writes exactly the count the scan reserved.
const SHADER: &str = r#"
struct Params {
    n: u32,
    table_mask: u32,   // table_size = table_mask + 1 (power of two)
    cell: f32,         // effective bucket edge (capped: max(cell, radius/K))
    radius: f32,
    radius2: f32,      // radius*radius, precomputed in f32 to match the gather test
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       pos: array<f32>;         // 3*n, xyz interleaved
@group(0) @binding(2) var<storage, read_write> slot_count: array<u32>;  // table_size
@group(0) @binding(3) var<storage, read_write> cursor: array<u32>;      // table_size
@group(0) @binding(4) var<storage, read_write> cell_start: array<u32>;  // table_size + 1
@group(0) @binding(5) var<storage, read_write> sorted_idx: array<u32>;  // n
@group(0) @binding(6) var<storage, read_write> nbr_count: array<u32>;   // n
@group(0) @binding(7) var<storage, read>       starts: array<u32>;      // n + 1 (host scan)
@group(0) @binding(8) var<storage, read_write> flat: array<u32>;        // total candidates

fn pos_of(i: u32) -> vec3<f32> {
    return vec3<f32>(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]);
}

// floor(p / cell) per axis. floor first, THEN cast, so negative coords bin like the
// CPU's `.floor() as i64` (a bare i32() cast would truncate toward zero).
fn cell_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p / params.cell));
}

// Hash a cell coord into a table slot. bitcast (not value-convert) so the 2's-complement
// bits of negative coords are used identically in build and query.
fn hash_cell(c: vec3<i32>) -> u32 {
    let ux = bitcast<u32>(c.x);
    let uy = bitcast<u32>(c.y);
    let uz = bitcast<u32>(c.z);
    let h = ux * 1640531513u ^ uy * 2654435789u ^ uz * 2246822519u;
    return h & params.table_mask;
}

// Single-invocation counting sort: histogram of slots, exclusive scan to bucket
// bases, then a stable serial scatter (ascending source index) into `sorted_idx`.
@compute @workgroup_size(1)
fn build() {
    let n = params.n;
    let ts = params.table_mask + 1u;
    for (var s = 0u; s < ts; s = s + 1u) {
        slot_count[s] = 0u;
        cursor[s] = 0u;
    }
    for (var i = 0u; i < n; i = i + 1u) {
        let slot = hash_cell(cell_of(pos_of(i)));
        slot_count[slot] = slot_count[slot] + 1u;
    }
    var running = 0u;
    for (var s = 0u; s < ts; s = s + 1u) {
        cell_start[s] = running;
        running = running + slot_count[s];
    }
    cell_start[ts] = running; // == n
    for (var i = 0u; i < n; i = i + 1u) {
        let slot = hash_cell(cell_of(pos_of(i)));
        let p = cell_start[slot] + cursor[slot];
        cursor[slot] = cursor[slot] + 1u;
        sorted_idx[p] = i;
    }
}

// Walk the box of cells covering pos[i] ± radius; for each candidate j, accept it
// only under its OWN cell (dedup: a j reached via two colliding cells is counted
// once, and a far flier colliding into a near slot is rejected here) and only if the
// true squared distance is within radius². Counts, or fills `flat[base..]`.
fn gather(i: u32, base: u32, fill: bool) -> u32 {
    let pi = pos_of(i);
    let lo = cell_of(pi - vec3<f32>(params.radius));
    let hi = cell_of(pi + vec3<f32>(params.radius));
    var cnt = 0u;
    for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
                let slot = hash_cell(vec3<i32>(cx, cy, cz));
                let s0 = cell_start[slot];
                let s1 = cell_start[slot + 1u];
                for (var p = s0; p < s1; p = p + 1u) {
                    let j = sorted_idx[p];
                    let cj = cell_of(pos_of(j));
                    if (cj.x == cx && cj.y == cy && cj.z == cz) {
                        let d = pos_of(j) - pi;
                        if (dot(d, d) <= params.radius2) {
                            if (fill) { flat[base + cnt] = j; }
                            cnt = cnt + 1u;
                        }
                    }
                }
            }
        }
    }
    return cnt;
}

@compute @workgroup_size(256)
fn query_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    nbr_count[i] = gather(i, 0u, false);
}

@compute @workgroup_size(256)
fn query_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    _ = gather(i, starts[i], true);
}
"#;

/// Uniform for the build/query kernels: particle count, hash-table mask, the (capped)
/// cell edge, the query radius, and radius² — the last three f32 to match the f32
/// device gather. Mirrors the WGSL `Params` (32-byte, 16-aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    table_mask: u32,
    cell: f32,
    radius: f32,
    radius2: f32,
    _pad: [u32; 3],
}

/// Local group size for the per-target query passes (mirrors the per-particle passes
/// in [`crate::GpuResidentLeapfrog`]).
const QUERY_WG: u32 = 256;

/// GPU Green-style spatial-hash neighbor search. Reusable wgpu compute context
/// built once ([`new`](Self::new)) and driven per [`query_all`](Self::query_all);
/// storage buffers grow lazily with N — the same bring-up idiom as
/// [`crate::GpuSorter`].
pub struct GpuNeighborGrid {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline_build: wgpu::ComputePipeline,
    pipeline_count: wgpu::ComputePipeline,
    pipeline_fill: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    // Storage grown lazily to the largest (n, table_size) seen. flat/flat_readback
    // grow independently to the largest candidate total seen.
    pos_buf: Option<wgpu::Buffer>,
    slot_count_buf: Option<wgpu::Buffer>,
    cursor_buf: Option<wgpu::Buffer>,
    cell_start_buf: Option<wgpu::Buffer>,
    sorted_idx_buf: Option<wgpu::Buffer>,
    nbr_count_buf: Option<wgpu::Buffer>,
    starts_buf: Option<wgpu::Buffer>,
    nbr_readback: Option<wgpu::Buffer>,
    flat_buf: Option<wgpu::Buffer>,
    flat_readback: Option<wgpu::Buffer>,
    capacity_n: usize,
    table_size: u32,
    flat_capacity: u32,
}

impl GpuNeighborGrid {
    /// Bring up a headless wgpu compute device and build the spatial-hash pipelines.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available,
    /// exactly like [`crate::GpuSorter::new`].
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
                label: Some("galaxy-gpu-sph-grid-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-sph-grid-shader"),
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
            label: Some("gpu-sph-grid-bgl"),
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
                storage(1, true),  // pos (read)
                storage(2, false), // slot_count (rw)
                storage(3, false), // cursor (rw)
                storage(4, false), // cell_start (rw)
                storage(5, false), // sorted_idx (rw)
                storage(6, false), // nbr_count (rw)
                storage(7, true),  // starts (read)
                storage(8, false), // flat (rw)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-sph-grid-pl"),
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
        let pipeline_build = make_pipeline("build", "gpu-sph-grid-build");
        let pipeline_count = make_pipeline("query_count", "gpu-sph-grid-count");
        let pipeline_fill = make_pipeline("query_fill", "gpu-sph-grid-fill");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-grid-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuNeighborGrid {
            device,
            queue,
            bgl,
            pipeline_build,
            pipeline_count,
            pipeline_fill,
            params_buf,
            pos_buf: None,
            slot_count_buf: None,
            cursor_buf: None,
            cell_start_buf: None,
            sorted_idx_buf: None,
            nbr_count_buf: None,
            starts_buf: None,
            nbr_readback: None,
            flat_buf: None,
            flat_readback: None,
            capacity_n: 0,
            table_size: 0,
            flat_capacity: 0,
        })
    }

    /// Ensure the per-particle / per-table storage holds at least `n` bodies and a
    /// hash table of `table_size` slots, reallocating (only grows) when either
    /// outgrows the current capacity. Caller guarantees `n > 0`.
    fn ensure_capacity(&mut self, n: usize, table_size: u32) {
        if n <= self.capacity_n && table_size <= self.table_size && self.pos_buf.is_some() {
            return;
        }
        let u32s = |count: u64, label: &str, extra: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: count * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let n64 = n as u64;
        let ts64 = table_size as u64;
        self.pos_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-grid-pos"),
            size: 3 * n64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.slot_count_buf = Some(u32s(
            ts64,
            "gpu-sph-grid-slot-count",
            wgpu::BufferUsages::empty(),
        ));
        self.cursor_buf = Some(u32s(
            ts64,
            "gpu-sph-grid-cursor",
            wgpu::BufferUsages::empty(),
        ));
        self.cell_start_buf = Some(u32s(
            ts64 + 1,
            "gpu-sph-grid-cell-start",
            wgpu::BufferUsages::empty(),
        ));
        self.sorted_idx_buf = Some(u32s(
            n64,
            "gpu-sph-grid-sorted-idx",
            wgpu::BufferUsages::empty(),
        ));
        self.nbr_count_buf = Some(u32s(
            n64,
            "gpu-sph-grid-nbr-count",
            wgpu::BufferUsages::COPY_SRC,
        ));
        self.starts_buf = Some(u32s(
            n64 + 1,
            "gpu-sph-grid-starts",
            wgpu::BufferUsages::COPY_DST,
        ));
        self.nbr_readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-grid-nbr-readback"),
            size: n64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.capacity_n = n;
        self.table_size = table_size;
    }

    /// Ensure the flat candidate buffer + its readback hold at least `total` entries.
    /// `total` may be 0 (all queries empty); a 1-entry floor keeps the storage binding
    /// non-zero-sized.
    fn ensure_flat(&mut self, total: u32) {
        let want = total.max(1);
        if want <= self.flat_capacity && self.flat_buf.is_some() {
            return;
        }
        let bytes = want as u64 * std::mem::size_of::<u32>() as u64;
        self.flat_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-grid-flat"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.flat_readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-grid-flat-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.flat_capacity = want;
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        let e = "buffers ensured";
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-sph-grid-bind-group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pos_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.slot_count_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cursor_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cell_start_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.sorted_idx_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.nbr_count_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.starts_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.flat_buf.as_ref().expect(e).as_entire_binding(),
                },
            ],
        })
    }

    /// Build the spatial hash over `pos` with cell edge `cell`, then return, for
    /// every particle `i`, the candidate indices `j` with `|pos[j] − pos[i]| ≤
    /// radius` (including `i`). `cell` and `radius` are decoupled: `cell` sizes the
    /// buckets and `radius` sizes the neighborhood walk — the wide-`h` regime is
    /// `radius ≫ cell`. The bucket edge is internally capped at `max(cell, radius/4)`
    /// so the walk stays ≤ 4 cells per axis (correctness-neutral; see the module
    /// docs and D4). `cell > 0`, finite.
    pub fn query_all(&mut self, pos: &[DVec3], cell: f64, radius: f64) -> GpuNeighbours {
        assert!(
            cell.is_finite() && cell > 0.0,
            "GpuNeighborGrid cell must be positive and finite, got {cell}"
        );
        let n = pos.len();
        if n == 0 {
            return GpuNeighbours {
                starts: vec![0],
                flat: Vec::new(),
            };
        }

        let table_size = table_size_for(n);
        self.ensure_capacity(n, table_size);
        // At least one candidate reservation so the bind group's flat binding is valid
        // during build/count (its contents are unused until the fill pass).
        self.ensure_flat(1);

        // Capped bucket edge: bounds the walk to ≤ WALK_CAP cells per axis.
        let cell_eff = cell.max(radius / WALK_CAP);
        let radius_f = radius as f32;
        let params = Params {
            n: n as u32,
            table_mask: table_size - 1,
            cell: cell_eff as f32,
            radius: radius_f,
            radius2: radius_f * radius_f,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Upload positions as interleaved f32 (physics is f64; the device is f32).
        let pos_f32: Vec<f32> = pos
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.queue.write_buffer(
            self.pos_buf.as_ref().expect("ensured"),
            0,
            bytemuck::cast_slice(&pos_f32),
        );

        // Pass 1: build the hash (single invocation) + count neighbors (per target).
        let bg = self.bind_group();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-grid-build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-grid-count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_count);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(QUERY_WG as usize) as u32, 1, 1);
        }
        let count_bytes = n as u64 * std::mem::size_of::<u32>() as u64;
        enc.copy_buffer_to_buffer(
            self.nbr_count_buf.as_ref().expect("ensured"),
            0,
            self.nbr_readback.as_ref().expect("ensured"),
            0,
            count_bytes,
        );
        self.queue.submit([enc.finish()]);
        let counts = self.read_u32(self.nbr_readback.as_ref().expect("ensured"), n);

        // Host exclusive scan → CSR row offsets. total = starts[n].
        let mut starts = Vec::with_capacity(n + 1);
        let mut running = 0u32;
        for &c in &counts {
            starts.push(running);
            running += c;
        }
        starts.push(running);
        let total = running;

        if total == 0 {
            return GpuNeighbours {
                starts,
                flat: Vec::new(),
            };
        }

        // Pass 2: fill. Upload starts, grow flat, re-walk and write each slice.
        self.ensure_flat(total);
        self.queue.write_buffer(
            self.starts_buf.as_ref().expect("ensured"),
            0,
            bytemuck::cast_slice(&starts),
        );
        // flat may have been reallocated → rebuild the bind group with the new buffer.
        let bg = self.bind_group();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-grid-fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_fill);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(QUERY_WG as usize) as u32, 1, 1);
        }
        let flat_bytes = total as u64 * std::mem::size_of::<u32>() as u64;
        enc.copy_buffer_to_buffer(
            self.flat_buf.as_ref().expect("ensured"),
            0,
            self.flat_readback.as_ref().expect("ensured"),
            0,
            flat_bytes,
        );
        self.queue.submit([enc.finish()]);
        let flat = self.read_u32(
            self.flat_readback.as_ref().expect("ensured"),
            total as usize,
        );

        GpuNeighbours { starts, flat }
    }

    /// Map a readback buffer and copy out its first `count` u32s. A map failure here
    /// is a genuine GPU loss (new() validated the device), so it panics rather than
    /// silently corrupting — the same discipline as [`crate::GpuSorter::sort`].
    fn read_u32(&self, readback: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let bytes = count as u64 * std::mem::size_of::<u32>() as u64;
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
        let out = bytemuck::cast_slice::<u8, u32>(&data)[..count].to_vec();
        drop(data);
        readback.unmap();
        out
    }
}

/// Hash-table size for `n` bodies: the next power of two ≥ `2n` (a load factor ≤ ½
/// keeps collisions cheap), floored at 64. Power-of-two so the slot reduction is a
/// mask (`& table_mask`).
fn table_size_for(n: usize) -> u32 {
    let target = (2 * n).max(64) as u32;
    target.next_power_of_two()
}
