//! [`GpuDensity`]: GPU adaptive-h SPH density (GPU-SPH G2).
//!
//! Per-gas-particle smoothing length `h` and density `ρ`, computed on the GPU by the
//! same root-find as the CPU oracle [`galaxy_solvers::sph::density_adaptive`]: `h`
//! solves `N_i(h) = n_ngb` where the kernel-weighted count
//! `N_i(h) = (4π/3)(SUPPORT·h)³ · Σ_j W(|x_i − x_j|, h)` is monotone in `h`, then
//! `ρ_i = Σ_j m_j W(|x_i − x_j|, h_i)`.
//!
//! ## GPU-side grid walk, no host CSR (reuses G1's hash/bin math)
//! The bisection walks the same Green-style counting-sort spatial hash as
//! [`crate::GpuNeighborGrid`] — this shader concatenates G1's
//! [`crate::sph_grid::GRID_HELPERS_WGSL`] (`pos_of`/`cell_of`/`hash_cell`/`build`),
//! so the bucketing is a single source of truth — but consumes the buckets
//! **GPU-side** (each thread re-walks the neighborhood per trial `h`), never a host
//! CSR round-trip. That is the shape the resident stepper (G5) needs.
//!
//! ## Per-particle radius ⇒ NOT walk-cappable (D4)
//! Unlike G1's fixed radius, the walk radius `SUPPORT·h` is per-particle, so the
//! build cell is fixed and the walk spans `ceil(SUPPORT·h/cell)` cells — unbounded as
//! `h` grows. There is no free walk cap here (coarsening the cell just over-gathers
//! the dense core instead); this is exactly why the endpoint is the LBVH (D4). The
//! grid mirrors the CPU (`cell ≈ SUPPORT·h_seed`) and is used at the modest,
//! measured-regime scale the gates cover. A `MAX_SPAN` backstop turns a would-be
//! runaway walk (only reachable via a non-rooted `h` blow-up) into a bounded,
//! debuggable wrong answer rather than a GPU hang.
//!
//! ## Precision & determinism
//! Positions/kernel are f32 (D1): the gate is f32-tolerance vs the f64 oracle, never
//! bit-exact. `ρ = Σ m_j W` and `N = pre·Σ W` are plain f32 accumulations (no
//! error-free-transform), so — unlike the DS-carried position drift — no XOR-barrier
//! is needed (D3 not triggered). Each thread owns its own `(ρ, h)` output slot and
//! the walk is a gather (no scatter race) → bit-identical run-to-run on a device.

use bytemuck::{Pod, Zeroable};

use galaxy_core::DVec3;

use crate::sph_grid::GRID_HELPERS_WGSL;
use crate::GpuError;

/// Kernel support radius in units of `h` (`W = 0` for `r ≥ SUPPORT·h`); the Monaghan
/// M4 convention, matching [`galaxy_solvers::sph::SUPPORT`]. Hardcoded (not imported)
/// to keep the WGSL literal and this host constant a matched pair, as in `sph_grid`.
const SUPPORT: f64 = 2.0;
const PI: f64 = std::f64::consts::PI;

/// Local group size for the per-target density pass (mirrors the resident stepper).
const QUERY_WG: u32 = 256;

/// Per-particle SPH density and the smoothing length it was computed at. `f32`
/// because the device computes in f32 (the CPU oracle is f64 and the gate is an
/// f32-tolerance comparison — D1/D5, never bit-exact).
pub struct DensityField {
    /// Density `ρ_i` per particle.
    pub rho: Vec<f32>,
    /// Adaptive smoothing length `h_i` per particle.
    pub h: Vec<f32>,
}

/// Density kernels: the cubic-spline `W`, the centered neighborhood sum, and the two
/// per-target passes (adaptive root-find / fixed-`h` summation). Concatenated after
/// [`GRID_HELPERS_WGSL`], which supplies `pos_of`/`cell_of`/`hash_cell`/`build`.
pub(crate) const DENSITY_DECLS: &str = r#"
struct Params {
    n: u32,
    table_mask: u32,   // table_size = table_mask + 1 (power of two)
    cell: f32,         // build/walk bucket edge (SUPPORT·h_seed adaptive; SUPPORT·h_max fixed)
    n_ngb: f32,        // target kernel-weighted neighbor count
    h_tol_rel: f32,    // bisection convergence: relative tolerance on h
    h_seed: f32,       // global bracket seed (rooted ⇒ h is seed-independent)
    h_cap: f32,        // rootless clamp / expand ceiling
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       pos: array<f32>;         // 3*n, xyz interleaved
@group(0) @binding(2) var<storage, read>       mass: array<f32>;        // n
@group(0) @binding(3) var<storage, read_write> slot_count: array<u32>;  // table_size
@group(0) @binding(4) var<storage, read_write> cursor: array<u32>;      // table_size
@group(0) @binding(5) var<storage, read_write> cell_start: array<u32>;  // table_size + 1
@group(0) @binding(6) var<storage, read_write> sorted_idx: array<u32>;  // n
@group(0) @binding(7) var<storage, read_write> h_io: array<f32>;        // n: OUT (adaptive) / IN (fixed)
@group(0) @binding(8) var<storage, read_write> rho_out: array<f32>;     // n
"#;

pub(crate) const DENSITY_KERNELS: &str = r#"
const SUPPORT: f32 = 2.0;
const PI: f32 = 3.1415926535897931;
// TDR backstop: a rooted particle's walk is ≤ a few cells; only a non-rooted h blow-up
// could drive SUPPORT·h/cell huge. Clip the span so that becomes a bounded wrong answer
// (caught by the gate), never a GPU hang. A clamped particle's true neighbors sit in
// its own/adjacent cells, so the clip is loss-free there too.
const MAX_SPAN: i32 = 32;

// Cubic-spline (M4) kernel W(r, h); matches galaxy_solvers::sph::kernel::w.
fn kernel_w(r: f32, h: f32) -> f32 {
    let q = r / h;
    let norm = 1.0 / (PI * h * h * h);
    if (q < 1.0) {
        return norm * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    } else if (q < 2.0) {
        let t = 2.0 - q;
        return norm * 0.25 * t * t * t;
    }
    return 0.0;
}

// Σ over neighbors within SUPPORT*h of (W, mass·W). The walk is CENTERED on the
// target's own cell and spans ceil(SUPPORT*h/cell)+1 cells each way — so the self
// cell is always covered (a lo..hi box can clip it when h is huge) — and cell-match +
// true-distance filter as in G1's gather (dedups hash collisions, rejects far bucket
// mates). far cells contribute exactly 0 (W = 0 past the support).
fn neighbor_sums(i: u32, h: f32) -> vec2<f32> {
    let pi = pos_of(i);
    let rad = SUPPORT * h;
    let rad2 = rad * rad;
    let c0 = cell_of(pi);
    var span = i32(ceil(rad / params.cell)) + 1;
    if (span > MAX_SPAN) { span = MAX_SPAN; }
    var sw = 0.0;
    var smw = 0.0;
    for (var dx = -span; dx <= span; dx = dx + 1) {
        for (var dy = -span; dy <= span; dy = dy + 1) {
            for (var dz = -span; dz <= span; dz = dz + 1) {
                let cx = c0.x + dx;
                let cy = c0.y + dy;
                let cz = c0.z + dz;
                let slot = hash_cell(vec3<i32>(cx, cy, cz));
                let s0 = cell_start[slot];
                let s1 = cell_start[slot + 1u];
                for (var p = s0; p < s1; p = p + 1u) {
                    let j = sorted_idx[p];
                    let cj = cell_of(pos_of(j));
                    if (cj.x == cx && cj.y == cy && cj.z == cz) {
                        let d = pos_of(j) - pi;
                        let r2 = dot(d, d);
                        if (r2 <= rad2) {
                            let wv = kernel_w(sqrt(r2), h);
                            sw = sw + wv;
                            smw = smw + mass[j] * wv;
                        }
                    }
                }
            }
        }
    }
    return vec2<f32>(sw, smw);
}

// N(h) = (4π/3)(SUPPORT·h)³ · Σ W — the monotone count the bisection roots.
fn count_of(h: f32, sw: f32) -> f32 {
    let sh = SUPPORT * h;
    return (4.0 * PI / 3.0) * sh * sh * sh * sw;
}

// Per-target adaptive-h: bracket from the global seed, expand to bound the root,
// bisect to h_tol_rel; rootless particles clamp deterministically (cap / shrink
// floor). Mirrors density_adaptive's solve_one; the unique root makes it
// seed-independent for rooted particles (gated).
@compute @workgroup_size(256)
fn density_adaptive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let ngb = params.n_ngb;
    let cap = params.h_cap;
    var lo = min(params.h_seed * 0.5, cap);
    var hi = min(params.h_seed * 2.0, cap);

    var up = 0u;
    loop {
        if (count_of(hi, neighbor_sums(i, hi).x) >= ngb || hi >= cap || up >= 64u) { break; }
        hi = min(hi * 2.0, cap);
        up = up + 1u;
    }
    var dn = 0u;
    loop {
        if (count_of(lo, neighbor_sums(i, lo).x) <= ngb || dn >= 60u) { break; }
        lo = lo * 0.5;
        dn = dn + 1u;
    }

    var h: f32;
    if (count_of(hi, neighbor_sums(i, hi).x) < ngb) {
        h = hi;
    } else if (count_of(lo, neighbor_sums(i, lo).x) > ngb) {
        h = lo;
    } else {
        var bg = 0u;
        loop {
            if (hi - lo <= params.h_tol_rel * hi || bg >= 100u) { break; }
            let mid = 0.5 * (lo + hi);
            if (count_of(mid, neighbor_sums(i, mid).x) < ngb) { lo = mid; } else { hi = mid; }
            bg = bg + 1u;
        }
        h = 0.5 * (lo + hi);
    }

    h_io[i] = h;
    rho_out[i] = neighbor_sums(i, h).y;
}

// Per-target fixed-h summation: ρ = Σ m_j W at the caller's h_io[i] (no root-find).
@compute @workgroup_size(256)
fn density_fixed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    rho_out[i] = neighbor_sums(i, h_io[i]).y;
}
"#;

/// Uniform for the density kernels; mirrors the WGSL `Params` (32-byte, 16-aligned).
/// The first three fields match G1's `Params` prefix so the shared `build` kernel
/// (from [`GRID_HELPERS_WGSL`]) compiles unchanged here.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Params {
    n: u32,
    table_mask: u32,
    cell: f32,
    n_ngb: f32,
    h_tol_rel: f32,
    h_seed: f32,
    h_cap: f32,
    _pad: u32,
}

/// Compute the adaptive-h density [`Params`] (bracket seed / cap / grid cell) from the
/// gas positions — the single source of truth for the root-find's seeding, shared by
/// [`GpuDensity::densities`] and the GPU-resident stepper (which fixes these at upload,
/// leveraging the root's seed-independence — G2). `n_ngb`/`h_tol_rel` come from the
/// caller's [`DensityConfig`]. Caller guarantees `pos` is non-empty.
pub(crate) fn density_params(pos: &[DVec3], n_ngb: f64, h_tol_rel: f64) -> Params {
    let n = pos.len();
    // Global spacing estimate → bracket seed and cap (mirrors density_adaptive).
    let (mut lo_c, mut hi_c) = (pos[0], pos[0]);
    for p in pos {
        lo_c = lo_c.min(*p);
        hi_c = hi_c.max(*p);
    }
    let extent = hi_c - lo_c;
    let diag = extent.length();
    let vol = extent.x * extent.y * extent.z;
    let s_est = if vol > 0.0 {
        (vol / n as f64).cbrt()
    } else if diag > 0.0 {
        diag / (n as f64).cbrt()
    } else {
        1.0
    };
    let h_seed = s_est * (3.0 * n_ngb / (32.0 * PI)).cbrt();
    let h_cap = (64.0 * h_seed).max(4.0 * diag);
    let cell = (SUPPORT * h_seed).max(1e-12);
    let table_size = table_size_for(n);
    Params {
        n: n as u32,
        table_mask: table_size - 1,
        cell: cell as f32,
        n_ngb: n_ngb as f32,
        h_tol_rel: h_tol_rel as f32,
        h_seed: h_seed as f32,
        h_cap: h_cap as f32,
        _pad: 0,
    }
}

/// GPU adaptive-h density. Reusable wgpu compute context built once ([`new`](Self::new))
/// and driven per call; storage grows lazily with N — the same bring-up idiom as
/// [`crate::GpuNeighborGrid`].
pub struct GpuDensity {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline_build: wgpu::ComputePipeline,
    pipeline_adaptive: wgpu::ComputePipeline,
    pipeline_fixed: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    pos_buf: Option<wgpu::Buffer>,
    mass_buf: Option<wgpu::Buffer>,
    slot_count_buf: Option<wgpu::Buffer>,
    cursor_buf: Option<wgpu::Buffer>,
    cell_start_buf: Option<wgpu::Buffer>,
    sorted_idx_buf: Option<wgpu::Buffer>,
    h_io_buf: Option<wgpu::Buffer>,
    rho_out_buf: Option<wgpu::Buffer>,
    h_readback: Option<wgpu::Buffer>,
    rho_readback: Option<wgpu::Buffer>,
    capacity_n: usize,
    table_size: u32,
}

impl GpuDensity {
    /// Bring up a headless wgpu compute device and the build + density pipelines.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available,
    /// exactly like [`crate::GpuNeighborGrid::new`].
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("galaxy-gpu-sph-density-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let src = format!("{DENSITY_DECLS}{GRID_HELPERS_WGSL}{DENSITY_KERNELS}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-sph-density-shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
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
            label: Some("gpu-sph-density-bgl"),
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
                storage(2, true),  // mass (read)
                storage(3, false), // slot_count (rw)
                storage(4, false), // cursor (rw)
                storage(5, false), // cell_start (rw)
                storage(6, false), // sorted_idx (rw)
                storage(7, false), // h_io (rw)
                storage(8, false), // rho_out (rw)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-sph-density-pl"),
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
        let pipeline_build = make_pipeline("build", "gpu-sph-density-build");
        let pipeline_adaptive = make_pipeline("density_adaptive", "gpu-sph-density-adaptive");
        let pipeline_fixed = make_pipeline("density_fixed", "gpu-sph-density-fixed");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-density-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuDensity {
            device,
            queue,
            bgl,
            pipeline_build,
            pipeline_adaptive,
            pipeline_fixed,
            params_buf,
            pos_buf: None,
            mass_buf: None,
            slot_count_buf: None,
            cursor_buf: None,
            cell_start_buf: None,
            sorted_idx_buf: None,
            h_io_buf: None,
            rho_out_buf: None,
            h_readback: None,
            rho_readback: None,
            capacity_n: 0,
            table_size: 0,
        })
    }

    /// Ensure the per-particle / per-table storage holds at least `n` bodies and a
    /// hash table of `table_size` slots, reallocating (only grows) when either
    /// outgrows the current capacity. Caller guarantees `n > 0`.
    fn ensure_capacity(&mut self, n: usize, table_size: u32) {
        if n <= self.capacity_n && table_size <= self.table_size && self.pos_buf.is_some() {
            return;
        }
        let n64 = n as u64;
        let ts64 = table_size as u64;
        let f32s = |count: u64, label: &str, extra: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: count * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let u32s = |count: u64, label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: count * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let readback = |count: u64, label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: count * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        self.pos_buf = Some(f32s(
            3 * n64,
            "gpu-sph-density-pos",
            wgpu::BufferUsages::COPY_DST,
        ));
        self.mass_buf = Some(f32s(
            n64,
            "gpu-sph-density-mass",
            wgpu::BufferUsages::COPY_DST,
        ));
        self.slot_count_buf = Some(u32s(ts64, "gpu-sph-density-slot-count"));
        self.cursor_buf = Some(u32s(ts64, "gpu-sph-density-cursor"));
        self.cell_start_buf = Some(u32s(ts64 + 1, "gpu-sph-density-cell-start"));
        self.sorted_idx_buf = Some(u32s(n64, "gpu-sph-density-sorted-idx"));
        self.h_io_buf = Some(f32s(
            n64,
            "gpu-sph-density-h-io",
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        ));
        self.rho_out_buf = Some(f32s(
            n64,
            "gpu-sph-density-rho-out",
            wgpu::BufferUsages::COPY_SRC,
        ));
        self.h_readback = Some(readback(n64, "gpu-sph-density-h-readback"));
        self.rho_readback = Some(readback(n64, "gpu-sph-density-rho-readback"));
        self.capacity_n = n;
        self.table_size = table_size;
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        let e = "buffers ensured";
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-sph-density-bind-group"),
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
                    resource: self.mass_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.slot_count_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cursor_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cell_start_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.sorted_idx_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.h_io_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.rho_out_buf.as_ref().expect(e).as_entire_binding(),
                },
            ],
        })
    }

    /// Adaptive-h density: per particle, bisect `h` to hit `n_ngb`, then sum `ρ`.
    /// `n_ngb` must exceed the self-term floor `32/3` (else no root exists for any
    /// cloud). The global bracket seed / cap are computed here (CPU-parity formulas);
    /// for a fully-rooted cloud the unique root makes `h` seed-independent (gated).
    pub fn densities(
        &mut self,
        pos: &[DVec3],
        mass: &[f64],
        n_ngb: f64,
        h_tol_rel: f64,
    ) -> DensityField {
        assert!(
            n_ngb > 32.0 / 3.0,
            "n_ngb must exceed the self-term floor 32/3, got {n_ngb}"
        );
        let n = pos.len();
        if n == 0 {
            return DensityField {
                rho: Vec::new(),
                h: Vec::new(),
            };
        }

        let table_size = table_size_for(n);
        self.ensure_capacity(n, table_size);
        // Bracket seed / cap / grid cell — the single source of truth shared with the
        // GPU-resident stepper.
        let params = density_params(pos, n_ngb, h_tol_rel);
        self.upload_inputs(&params, pos, mass);
        let (rho, h) = self.dispatch(n, true);
        DensityField { rho, h }
    }

    /// Fixed-`h` density summation (the decoupled summation gate): `ρ_i = Σ_j m_j
    /// W(r_ij, h_i)` at the caller's `h`, no root-find. Builds the grid at
    /// `SUPPORT·h_max` (as [`galaxy_solvers::sph::density_fixed`] does).
    pub fn densities_at(&mut self, pos: &[DVec3], mass: &[f64], h: &[f64]) -> Vec<f32> {
        assert_eq!(pos.len(), h.len(), "pos and h length mismatch");
        let n = pos.len();
        if n == 0 {
            return Vec::new();
        }
        let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(
            h_max.is_finite() && h_max > 0.0,
            "densities_at needs positive finite smoothing lengths"
        );
        let cell = (SUPPORT * h_max).max(1e-12);

        let table_size = table_size_for(n);
        self.ensure_capacity(n, table_size);
        let params = Params {
            n: n as u32,
            table_mask: table_size - 1,
            cell: cell as f32,
            n_ngb: 0.0,
            h_tol_rel: 0.0,
            h_seed: 0.0,
            h_cap: 0.0,
            _pad: 0,
        };
        self.upload_inputs(&params, pos, mass);
        // The fixed path READS h from h_io — upload the caller's h (as f32).
        let hf: Vec<f32> = h.iter().map(|&x| x as f32).collect();
        self.queue.write_buffer(
            self.h_io_buf.as_ref().expect("ensured"),
            0,
            bytemuck::cast_slice(&hf),
        );
        let (rho, _h) = self.dispatch(n, false);
        rho
    }

    /// Upload params + interleaved f32 positions + f32 masses.
    fn upload_inputs(&self, params: &Params, pos: &[DVec3], mass: &[f64]) {
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
        let pos_f32: Vec<f32> = pos
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.queue.write_buffer(
            self.pos_buf.as_ref().expect("ensured"),
            0,
            bytemuck::cast_slice(&pos_f32),
        );
        let mass_f32: Vec<f32> = mass.iter().map(|&m| m as f32).collect();
        self.queue.write_buffer(
            self.mass_buf.as_ref().expect("ensured"),
            0,
            bytemuck::cast_slice(&mass_f32),
        );
    }

    /// Build the hash (single invocation) then run the density pass (per target),
    /// reading back `(ρ, h)`. `adaptive` selects the root-find vs the fixed-`h` sum.
    fn dispatch(&self, n: usize, adaptive: bool) -> (Vec<f32>, Vec<f32>) {
        let bg = self.bind_group();
        let pipeline = if adaptive {
            &self.pipeline_adaptive
        } else {
            &self.pipeline_fixed
        };
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-density-build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-density-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(QUERY_WG as usize) as u32, 1, 1);
        }
        let bytes = n as u64 * std::mem::size_of::<f32>() as u64;
        enc.copy_buffer_to_buffer(
            self.rho_out_buf.as_ref().expect("ensured"),
            0,
            self.rho_readback.as_ref().expect("ensured"),
            0,
            bytes,
        );
        enc.copy_buffer_to_buffer(
            self.h_io_buf.as_ref().expect("ensured"),
            0,
            self.h_readback.as_ref().expect("ensured"),
            0,
            bytes,
        );
        self.queue.submit([enc.finish()]);
        let rho = self.read_f32(self.rho_readback.as_ref().expect("ensured"), n);
        let h = self.read_f32(self.h_readback.as_ref().expect("ensured"), n);
        (rho, h)
    }

    /// Map a readback buffer and copy out its first `count` f32s. A map failure is a
    /// genuine GPU loss (new() validated the device), so it panics — same discipline
    /// as [`crate::GpuNeighborGrid`].
    fn read_f32(&self, readback: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let bytes = count as u64 * std::mem::size_of::<f32>() as u64;
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
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..count].to_vec();
        drop(data);
        readback.unmap();
        out
    }
}

/// Hash-table size for `n` bodies: the next power of two ≥ `2n`, floored at 64
/// (same policy as `sph_grid`). Power-of-two so the slot reduction is a mask.
pub(crate) fn table_size_for(n: usize) -> u32 {
    let target = (2 * n).max(64) as u32;
    target.next_power_of_two()
}
