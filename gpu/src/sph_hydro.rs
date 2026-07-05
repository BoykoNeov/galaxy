//! [`GpuHydro`]: GPU isothermal SPH hydro force (GPU-SPH G3).
//!
//! Per-gas-particle acceleration in the symmetric `P/ρ²` form with the kernel-average
//! symmetrization (D2) and Monaghan (1992) artificial viscosity, matching the CPU
//! oracle [`galaxy_solvers::sph::hydro_accelerations`]:
//!
//! ```text
//! a_i = −Σ_j m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) ∇_i W̄_ij,   W̄_ij = ½(W(r,h_i)+W(r,h_j))
//! ```
//!
//! `ρ`/`h` are INPUTS (the [`crate::GpuDensity`] pass ran first), so — unlike G2 — there
//! is no root-find and no clamp/all-rooted subtlety; `h` is bit-identical to whatever
//! the CPU oracle is fed.
//!
//! ## Gather radius = global `SUPPORT·h_max` (load-bearing, NOT per-target)
//! `W̄` is nonzero for `r < 2·max(h_i,h_j)`, so a pair with `2h_i < r < 2h_j` contributes
//! force to BOTH i and j. Gathering only `SUPPORT·h_i` per target would give i's force to
//! j but not j's to i, breaking Newton's third law and momentum conservation. So every
//! thread walks the SAME global `SUPPORT·h_max` radius (built into `cell` host-side),
//! exactly as the CPU oracle does. A pair with `2·max(h_i,h_j) < r < SUPPORT·h_max`
//! contributes exactly 0 (both `grad_w` terms vanish) — harmless, and it makes the
//! gather boundary force-invisible (no cross-device boundary-robustness worry).
//!
//! ## Determinism & precision (D1/D2/D3)
//! Gather-per-target: each thread owns its own `acc` slot (no scatter race) → bit-
//! identical run-to-run on a device. The force sum is a plain f32 vector accumulation
//! (NOT an error-free-transform), so no DS XOR-barrier is needed (D3 not triggered),
//! same as G2. With EQUAL mass each pair's two contributions are exact f32 negatives
//! (`grad_w(−r) = −grad_w(r)` exactly; `coeff` commutative-equal), so total momentum
//! drift is reduction roundoff only — the sharp antisymmetry gate rests on this.

use bytemuck::{Pod, Zeroable};

use galaxy_core::DVec3;
use galaxy_solvers::sph::HydroParams;

use crate::sph_grid::GRID_HELPERS_WGSL;
use crate::GpuError;

/// Kernel support radius in units of `h` (`W = 0` for `r ≥ SUPPORT·h`); matches
/// [`galaxy_solvers::sph::SUPPORT`]. Hardcoded (not imported) to keep the WGSL literal
/// and this host constant a matched pair, as in `sph_grid` / `sph_density`.
const SUPPORT: f64 = 2.0;

/// Local group size for the per-target hydro pass (mirrors the resident stepper).
const QUERY_WG: u32 = 256;

/// Hydro `Params` + bind-group declarations. The first three fields `{n, table_mask,
/// cell}` and the vars `pos`/`slot_count`/`cursor`/`cell_start`/`sorted_idx` match what
/// [`GRID_HELPERS_WGSL`]'s shared `build` expects, so that text compiles unchanged here.
const HYDRO_DECLS: &str = r#"
struct Params {
    n: u32,
    table_mask: u32,     // table_size = table_mask + 1 (power of two)
    cell: f32,           // build/walk bucket edge = SUPPORT·h_max (global gather radius)
    radius: f32,         // global gather radius = SUPPORT·h_max
    sound_speed: f32,    // isothermal c_s (EOS P = c_s²ρ)
    alpha: f32,          // Monaghan viscosity linear coeff
    beta: f32,           // Monaghan viscosity quadratic coeff
    visc_eps2: f32,      // μ-denominator regularization ε²
};

// NB: the default wgpu limit is 8 storage buffers/stage (the whole crate stays on
// `Limits::default()` for portability), so the three per-particle scalars mass/ρ/h are
// PACKED into one interleaved `scalars` buffer rather than three — keeping the count at
// exactly 8 (pos, vel, scalars + the four grid buffers + acc_out).
@group(0) @binding(0)  var<uniform>             params: Params;
@group(0) @binding(1)  var<storage, read>       pos: array<f32>;         // 3*n, xyz interleaved
@group(0) @binding(2)  var<storage, read>       vel: array<f32>;         // 3*n, xyz interleaved
@group(0) @binding(3)  var<storage, read>       scalars: array<f32>;     // 3*n: [mass, ρ, h] per particle
@group(0) @binding(4)  var<storage, read_write> slot_count: array<u32>;  // table_size
@group(0) @binding(5)  var<storage, read_write> cursor: array<u32>;      // table_size
@group(0) @binding(6)  var<storage, read_write> cell_start: array<u32>;  // table_size + 1
@group(0) @binding(7)  var<storage, read_write> sorted_idx: array<u32>;  // n
@group(0) @binding(8)  var<storage, read_write> acc_out: array<f32>;     // 3*n
"#;

const HYDRO_KERNELS: &str = r#"
const SUPPORT: f32 = 2.0;
const PI: f32 = 3.1415926535897931;
// TDR backstop (vestigial here — cell = radius so span ≈ 2 regardless of h_max; kept
// for parity with the density walk, which genuinely needs it under a runaway h).
const MAX_SPAN: i32 = 32;

fn vel_of(i: u32) -> vec3<f32> {
    return vec3<f32>(vel[3u * i], vel[3u * i + 1u], vel[3u * i + 2u]);
}

// Per-particle scalars, packed [mass, ρ, h] per particle (see the DECLS note).
fn mass_of(i: u32) -> f32 { return scalars[3u * i]; }
fn rho_of(i: u32) -> f32 { return scalars[3u * i + 1u]; }
fn h_of(i: u32) -> f32 { return scalars[3u * i + 2u]; }

// ∇_i W(|r_ij|, h) for the separation r_ij = x_i − x_j; matches galaxy_solvers grad_w.
// Zero at r = 0 (smooth origin) and outside the support. grad_w(−r,h) = −grad_w(r,h)
// EXACTLY in f32 (length is even; negation distributes), which is what makes the
// pairwise force exactly antisymmetric under equal mass.
fn kernel_grad_w(r_ij: vec3<f32>, hh: f32) -> vec3<f32> {
    let r = length(r_ij);
    let q = r / hh;
    if (r == 0.0 || q >= 2.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let norm = 1.0 / (PI * hh * hh * hh);
    var dp: f32;
    if (q < 1.0) {
        dp = -3.0 * q + 2.25 * q * q;
    } else {
        let t = 2.0 - q;
        dp = -0.75 * t * t;
    }
    return r_ij * (norm * dp / (hh * r));
}

// a_i = −Σ_j m_j (term_i + term_j + Π_ij) ∇_i W̄_ij, gathered per target over the global
// SUPPORT·h_max. The walk is CENTERED on the target's own cell (self cell always
// covered) and cell-match + true-distance filtered as in G1/G2. self (j==i) is skipped
// (grad_w(0)=0 makes it zero anyway; skipped to mirror the oracle exactly).
fn hydro_force(i: u32) -> vec3<f32> {
    let pi = pos_of(i);
    let vi = vel_of(i);
    let hi = h_of(i);
    let cs = params.sound_speed;
    let cs2 = cs * cs;
    let term_i = cs2 / rho_of(i);   // P_i/ρ_i² (isothermal)
    let rad = params.radius;
    let rad2 = rad * rad;
    let c0 = cell_of(pi);
    var span = i32(ceil(rad / params.cell)) + 1;
    if (span > MAX_SPAN) { span = MAX_SPAN; }
    var a = vec3<f32>(0.0, 0.0, 0.0);
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
                    if (j == i) { continue; }
                    let cj = cell_of(pos_of(j));
                    if (cj.x == cx && cj.y == cy && cj.z == cz) {
                        let d = pi - pos_of(j);      // r_ij = x_i − x_j
                        let r2 = dot(d, d);
                        if (r2 <= rad2) {
                            let hj = h_of(j);
                            // ∇_i W̄ = ½(∇W(h_i) + ∇W(h_j)); exact negation of ∇_j W̄_ji.
                            let grad_avg = (kernel_grad_w(d, hi) + kernel_grad_w(d, hj)) * 0.5;
                            let term_j = cs2 / rho_of(j);
                            let v_ij = vi - vel_of(j);
                            let vr = dot(v_ij, d);
                            var visc = 0.0;
                            if (vr < 0.0) {         // Monaghan viscosity, only on approach
                                let h_bar = 0.5 * (hi + hj);
                                let rho_bar = 0.5 * (rho_of(i) + rho_of(j));
                                let mu = h_bar * vr / (r2 + params.visc_eps2 * h_bar * h_bar);
                                // Isothermal: c̄ = c_s (constant sound speed).
                                visc = (-params.alpha * cs * mu + params.beta * mu * mu) / rho_bar;
                            }
                            let coeff = term_i + term_j + visc;
                            // −m_j·coeff·∇_i W̄ — structured so the equal-mass pair term is
                            // the exact f32 negation of particle j's contribution.
                            a = a + grad_avg * (-mass_of(j) * coeff);
                        }
                    }
                }
            }
        }
    }
    return a;
}

@compute @workgroup_size(256)
fn hydro_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let a = hydro_force(i);
    acc_out[3u * i] = a.x;
    acc_out[3u * i + 1u] = a.y;
    acc_out[3u * i + 2u] = a.z;
}
"#;

/// Uniform for the hydro kernels; mirrors the WGSL `Params` (32-byte, 16-aligned). The
/// first three fields match G1/G2's `Params` prefix so the shared `build` kernel (from
/// [`GRID_HELPERS_WGSL`]) compiles unchanged.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    table_mask: u32,
    cell: f32,
    radius: f32,
    sound_speed: f32,
    alpha: f32,
    beta: f32,
    visc_eps2: f32,
}

/// GPU isothermal hydro force. Reusable wgpu compute context built once
/// ([`new`](Self::new)) and driven per call; storage grows lazily with N — the same
/// bring-up idiom as [`crate::GpuDensity`].
pub struct GpuHydro {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline_build: wgpu::ComputePipeline,
    pipeline_hydro: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    pos_buf: Option<wgpu::Buffer>,
    vel_buf: Option<wgpu::Buffer>,
    scalars_buf: Option<wgpu::Buffer>,
    slot_count_buf: Option<wgpu::Buffer>,
    cursor_buf: Option<wgpu::Buffer>,
    cell_start_buf: Option<wgpu::Buffer>,
    sorted_idx_buf: Option<wgpu::Buffer>,
    acc_out_buf: Option<wgpu::Buffer>,
    acc_readback: Option<wgpu::Buffer>,
    capacity_n: usize,
    table_size: u32,
}

impl GpuHydro {
    /// Bring up a headless wgpu compute device and the build + hydro pipelines.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available,
    /// exactly like [`crate::GpuDensity::new`].
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
                label: Some("galaxy-gpu-sph-hydro-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let src = format!("{HYDRO_DECLS}{GRID_HELPERS_WGSL}{HYDRO_KERNELS}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-sph-hydro-shader"),
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
            label: Some("gpu-sph-hydro-bgl"),
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
                storage(2, true),  // vel (read)
                storage(3, true),  // scalars = [mass, ρ, h] (read)
                storage(4, false), // slot_count (rw)
                storage(5, false), // cursor (rw)
                storage(6, false), // cell_start (rw)
                storage(7, false), // sorted_idx (rw)
                storage(8, false), // acc_out (rw)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-sph-hydro-pl"),
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
        let pipeline_build = make_pipeline("build", "gpu-sph-hydro-build");
        let pipeline_hydro = make_pipeline("hydro_main", "gpu-sph-hydro-main");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-hydro-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuHydro {
            device,
            queue,
            bgl,
            pipeline_build,
            pipeline_hydro,
            params_buf,
            pos_buf: None,
            vel_buf: None,
            scalars_buf: None,
            slot_count_buf: None,
            cursor_buf: None,
            cell_start_buf: None,
            sorted_idx_buf: None,
            acc_out_buf: None,
            acc_readback: None,
            capacity_n: 0,
            table_size: 0,
        })
    }

    /// Per-gas-particle acceleration `a_i = −Σ_j m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) ∇_i W̄_ij`,
    /// gathering per target over the global `SUPPORT·h_max`. `rho`/`h` are supplied (the
    /// density pass ran first); every slice must have length `pos.len()`.
    pub fn accelerations(
        &mut self,
        pos: &[DVec3],
        vel: &[DVec3],
        mass: &[f64],
        rho: &[f64],
        h: &[f64],
        params: &HydroParams,
    ) -> Vec<DVec3> {
        let n = pos.len();
        assert_eq!(vel.len(), n, "vel length mismatch");
        assert_eq!(mass.len(), n, "mass length mismatch");
        assert_eq!(rho.len(), n, "rho length mismatch");
        assert_eq!(h.len(), n, "h length mismatch");
        if n == 0 {
            return Vec::new();
        }
        let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(
            h_max.is_finite() && h_max > 0.0,
            "GPU hydro needs positive finite smoothing lengths"
        );
        // Global gather radius = SUPPORT·h_max (load-bearing — see the module doc). The
        // build cell equals it, so the centered walk spans ≈2 cells and captures every
        // averaged-kernel neighbor of every target regardless of the h dynamic range.
        let radius = SUPPORT * h_max;
        let cell = radius.max(1e-12);

        let table_size = table_size_for(n);
        self.ensure_capacity(n, table_size);
        let gp = Params {
            n: n as u32,
            table_mask: table_size - 1,
            cell: cell as f32,
            radius: radius as f32,
            sound_speed: params.sound_speed as f32,
            alpha: params.alpha as f32,
            beta: params.beta as f32,
            visc_eps2: params.visc_eps2 as f32,
        };
        self.upload_inputs(&gp, pos, vel, mass, rho, h);
        self.dispatch(n)
    }

    /// Ensure per-particle / per-table storage holds at least `n` bodies and a hash
    /// table of `table_size` slots, reallocating (only grows) when either outgrows the
    /// current capacity. Caller guarantees `n > 0`.
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
        let dst = wgpu::BufferUsages::COPY_DST;
        self.pos_buf = Some(f32s(3 * n64, "gpu-sph-hydro-pos", dst));
        self.vel_buf = Some(f32s(3 * n64, "gpu-sph-hydro-vel", dst));
        self.scalars_buf = Some(f32s(3 * n64, "gpu-sph-hydro-scalars", dst));
        self.slot_count_buf = Some(u32s(ts64, "gpu-sph-hydro-slot-count"));
        self.cursor_buf = Some(u32s(ts64, "gpu-sph-hydro-cursor"));
        self.cell_start_buf = Some(u32s(ts64 + 1, "gpu-sph-hydro-cell-start"));
        self.sorted_idx_buf = Some(u32s(n64, "gpu-sph-hydro-sorted-idx"));
        self.acc_out_buf = Some(f32s(
            3 * n64,
            "gpu-sph-hydro-acc-out",
            wgpu::BufferUsages::COPY_SRC,
        ));
        self.acc_readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-hydro-acc-readback"),
            size: 3 * n64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.capacity_n = n;
        self.table_size = table_size;
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        let e = "buffers ensured";
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-sph-hydro-bind-group"),
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
                    resource: self.vel_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scalars_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.slot_count_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cursor_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.cell_start_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.sorted_idx_buf.as_ref().expect(e).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.acc_out_buf.as_ref().expect(e).as_entire_binding(),
                },
            ],
        })
    }

    /// Upload params + interleaved f32 positions/velocities + f32 masses/densities/h.
    fn upload_inputs(
        &self,
        params: &Params,
        pos: &[DVec3],
        vel: &[DVec3],
        mass: &[f64],
        rho: &[f64],
        h: &[f64],
    ) {
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
        let interleave = |v: &[DVec3]| -> Vec<f32> {
            v.iter()
                .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
                .collect()
        };
        let write = |buf: &Option<wgpu::Buffer>, data: &[f32]| {
            self.queue.write_buffer(
                buf.as_ref().expect("ensured"),
                0,
                bytemuck::cast_slice(data),
            );
        };
        write(&self.pos_buf, &interleave(pos));
        write(&self.vel_buf, &interleave(vel));
        // Pack the three per-particle scalars [mass, ρ, h] into one buffer (8-storage
        // limit) — must match the WGSL `mass_of`/`rho_of`/`h_of` accessors.
        let scalars: Vec<f32> = (0..mass.len())
            .flat_map(|i| [mass[i] as f32, rho[i] as f32, h[i] as f32])
            .collect();
        write(&self.scalars_buf, &scalars);
    }

    /// Build the hash (single invocation) then run the hydro pass (per target), reading
    /// back the 3·n interleaved accelerations as `DVec3`s.
    fn dispatch(&self, n: usize) -> Vec<DVec3> {
        let bg = self.bind_group();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-hydro-build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-hydro-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_hydro);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(QUERY_WG as usize) as u32, 1, 1);
        }
        let bytes = 3 * n as u64 * std::mem::size_of::<f32>() as u64;
        enc.copy_buffer_to_buffer(
            self.acc_out_buf.as_ref().expect("ensured"),
            0,
            self.acc_readback.as_ref().expect("ensured"),
            0,
            bytes,
        );
        self.queue.submit([enc.finish()]);

        let readback = self.acc_readback.as_ref().expect("ensured");
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
        let flat = bytemuck::cast_slice::<u8, f32>(&data);
        let out: Vec<DVec3> = flat[..3 * n]
            .chunks_exact(3)
            .map(|c| DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64))
            .collect();
        drop(data);
        readback.unmap();
        out
    }
}

/// Hash-table size for `n` bodies: the next power of two ≥ `2n`, floored at 64 (same
/// policy as `sph_grid` / `sph_density`). Power-of-two so the slot reduction is a mask.
fn table_size_for(n: usize) -> u32 {
    let target = (2 * n).max(64) as u32;
    target.next_power_of_two()
}
