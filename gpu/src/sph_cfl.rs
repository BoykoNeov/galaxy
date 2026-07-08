//! [`GpuCfl`]: GPU isothermal SPH CFL reduction (GPU-SPH G4).
//!
//! Per-gas-particle stable step, matching the CPU oracle [`galaxy_solvers::sph::max_stable_dt`]:
//!
//! ```text
//! dt_i = C_cfl · h_i / v_sig,i,   v_sig,i = max_j (2c_s − 3 w_ij)  over  w_ij < 0,  floored at 2c_s
//! ```
//!
//! with the Gadget-style projected signal velocity `w_ij = (v_i−v_j)·r̂_ij` (the sign
//! decides approach/recede). The milestone bound is `min_i dt_i` — D6's per-batch
//! adaptive-dt substrate. `h` is an INPUT (the [`crate::GpuDensity`] pass ran first), so
//! there is no root-find here; `h` is bit-identical to whatever the CPU oracle is fed.
//!
//! ## Gather radius = global `SUPPORT·h_max`, and the cutoff is EXPLICIT
//! The force law couples a pair out to `2·max(h_i,h_j)` (the averaged kernel `W̄` is
//! nonzero there), so a compact small-h_i particle can be driven by a diffuse large-h_j
//! approacher whose support reaches it even though `2·h_i < r`. `v_sig,i` must see that
//! approach, so every thread walks the SAME global `SUPPORT·h_max` radius — NOT a
//! per-target `SUPPORT·h_i`, which would miss the approacher and leave `v_sig` stuck at
//! the `2c_s` floor (overestimating the stable dt). Unlike the G3 hydro force there is
//! no `grad_w` to vanish past the coupling range, so the kernel REJECTS each pair
//! outside its own `SUPPORT·max(h_i,h_j)` with a hard `r >= coupling` test.
//!
//! ## Determinism & precision (D1/D2)
//! Per-target gather: each thread owns its own `dt_out` slot (no scatter race) → bit-
//! identical run-to-run on a device. `dt_i` is a single max-reduction over approachers
//! and one divide — NO accumulation — so the GPU-vs-oracle agreement is tight f32. The
//! milestone `min_i dt_i` is reduced on the host: f32 `min` is exact and order-
//! independent (the result is bit-for-bit one of the inputs), so the reduction carries
//! no numerics — the GPU-resident no-readback min is deferred to G5, where the resident
//! stepper's dt-threading defines its interface.

use bytemuck::{Pod, Zeroable};

use galaxy_core::DVec3;
use galaxy_solvers::sph::HydroParams;

use crate::sph_grid::GRID_HELPERS_WGSL;
use crate::GpuError;

/// Kernel support radius in units of `h`; matches [`galaxy_solvers::sph::SUPPORT`].
/// Hardcoded (not imported) to keep the WGSL literal and this host constant a matched
/// pair, as in `sph_grid` / `sph_density` / `sph_hydro`.
const SUPPORT: f64 = 2.0;

/// Local group size for the per-target CFL pass.
const QUERY_WG: u32 = 256;

/// CFL `Params` + bind-group declarations. The first three fields `{n, table_mask, cell}`
/// and the vars `pos`/`slot_count`/`cursor`/`cell_start`/`sorted_idx` match what
/// [`GRID_HELPERS_WGSL`]'s shared `build` expects, so that text compiles unchanged here.
/// `pub(crate)` so the GPU-resident stepper (GPU-SPH G5c) reuses this text VERBATIM —
/// one source of truth for the resident CFL pass, as the hydro force reuses `HYDRO_DECLS`.
pub(crate) const CFL_DECLS: &str = r#"
struct Params {
    n: u32,
    table_mask: u32,     // table_size = table_mask + 1 (power of two)
    cell: f32,           // build/walk bucket edge = SUPPORT·h_max (global gather radius)
    radius: f32,         // global gather radius = SUPPORT·h_max
    sound_speed: f32,    // isothermal c_s
    c_cfl: f32,          // CFL number (dt_i = c_cfl·h_i/v_sig,i)
    _pad0: f32,          // pad to 32 bytes (uniform struct 16-alignment)
    _pad1: f32,
};

// CFL needs neither mass nor ρ (only h, v, c_s, c_cfl), so the eight storage buffers are
// pos, vel, h + the four grid buffers + dt_out — no packing needed, and the whole crate
// stays on wgpu::Limits::default() (8 storage buffers/stage) for portability.
@group(0) @binding(0)  var<uniform>             params: Params;
@group(0) @binding(1)  var<storage, read>       pos: array<f32>;         // 3*n, xyz interleaved
@group(0) @binding(2)  var<storage, read>       vel: array<f32>;         // 3*n, xyz interleaved
@group(0) @binding(3)  var<storage, read>       h: array<f32>;           // n, smoothing lengths
@group(0) @binding(4)  var<storage, read_write> slot_count: array<u32>;  // table_size
@group(0) @binding(5)  var<storage, read_write> cursor: array<u32>;      // table_size
@group(0) @binding(6)  var<storage, read_write> cell_start: array<u32>;  // table_size + 1
@group(0) @binding(7)  var<storage, read_write> sorted_idx: array<u32>;  // n
@group(0) @binding(8)  var<storage, read_write> dt_out: array<f32>;      // n
"#;

/// The CFL kernels (`cfl_dt` + `cfl_main`), reused VERBATIM by the resident stepper (G5c).
pub(crate) const CFL_KERNELS: &str = r#"
const SUPPORT: f32 = 2.0;
// TDR backstop (vestigial here — cell = radius so span ≈ 2 regardless of h_max; kept
// for parity with the density walk, which genuinely needs it under a runaway h).
const MAX_SPAN: i32 = 32;

fn vel_of(i: u32) -> vec3<f32> {
    return vec3<f32>(vel[3u * i], vel[3u * i + 1u], vel[3u * i + 2u]);
}

// dt_i = c_cfl·h_i / v_sig,i, with v_sig,i = max over APPROACHING neighbors of
// (2c_s − 3 w_ij), floored at 2c_s. Gathered per target over the global SUPPORT·h_max;
// the walk is CENTERED on the target's own cell and cell-match + true-distance filtered
// as in G1–G3. The coupling cutoff is EXPLICIT (r >= SUPPORT·max(h_i,h_j) ⇒ skip): a
// pair between its own coupling range and the global gather radius must NOT drive v_sig.
fn cfl_dt(i: u32) -> f32 {
    let pi = pos_of(i);
    let vi = vel_of(i);
    let hi = h[i];
    let cs = params.sound_speed;
    let two_cs = 2.0 * cs;
    let rad = params.radius;
    let c0 = cell_of(pi);
    var span = i32(ceil(rad / params.cell)) + 1;
    if (span > MAX_SPAN) { span = MAX_SPAN; }
    var v_sig = two_cs;   // floor: nothing approaching ⇒ v_sig = 2c_s
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
                        let d = pi - pos_of(j);        // r_ij = x_i − x_j
                        let r = length(d);
                        let coupling = SUPPORT * max(hi, h[j]);
                        // EXPLICIT cutoff: outside the pair's force-coupling range ⇒ no
                        // drive (no grad_w to vanish here, unlike the G3 force).
                        if (r == 0.0 || r >= coupling) { continue; }
                        // w_ij = (v_i − v_j)·r̂_ij  — divide by r (length), not r².
                        let w = dot(vi - vel_of(j), d) / r;
                        if (w < 0.0) {                 // approaching ⇒ raises v_sig
                            v_sig = max(v_sig, two_cs - 3.0 * w);
                        }
                    }
                }
            }
        }
    }
    return params.c_cfl * hi / v_sig;
}

@compute @workgroup_size(256)
fn cfl_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    dt_out[i] = cfl_dt(i);
}
"#;

/// Uniform for the CFL kernels; mirrors the WGSL `Params` (32-byte, 16-aligned). The
/// first three fields match G1–G3's `Params` prefix so the shared `build` kernel (from
/// [`GRID_HELPERS_WGSL`]) compiles unchanged.
/// `pub(crate)` so the resident stepper (G5c) writes this same uniform layout when it
/// reuses [`CFL_DECLS`]/[`CFL_KERNELS`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Params {
    pub(crate) n: u32,
    pub(crate) table_mask: u32,
    pub(crate) cell: f32,
    pub(crate) radius: f32,
    pub(crate) sound_speed: f32,
    pub(crate) c_cfl: f32,
    pub(crate) _pad0: f32,
    pub(crate) _pad1: f32,
}

/// GPU isothermal CFL reduction. Reusable wgpu compute context built once
/// ([`new`](Self::new)) and driven per call; storage grows lazily with N — the same
/// bring-up idiom as [`crate::GpuHydro`].
pub struct GpuCfl {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline_build: wgpu::ComputePipeline,
    pipeline_cfl: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    pos_buf: Option<wgpu::Buffer>,
    vel_buf: Option<wgpu::Buffer>,
    h_buf: Option<wgpu::Buffer>,
    slot_count_buf: Option<wgpu::Buffer>,
    cursor_buf: Option<wgpu::Buffer>,
    cell_start_buf: Option<wgpu::Buffer>,
    sorted_idx_buf: Option<wgpu::Buffer>,
    dt_out_buf: Option<wgpu::Buffer>,
    dt_readback: Option<wgpu::Buffer>,
    capacity_n: usize,
    table_size: u32,
}

impl GpuCfl {
    /// Bring up a headless wgpu compute device and the build + CFL pipelines. Returns a
    /// typed [`GpuError`] (never panics) when no adapter is available, exactly like
    /// [`crate::GpuHydro::new`].
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let crate::context::GpuContext { device, queue } = crate::context::gpu_context()?;

        let src = format!("{CFL_DECLS}{GRID_HELPERS_WGSL}{CFL_KERNELS}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-sph-cfl-shader"),
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
            label: Some("gpu-sph-cfl-bgl"),
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
                storage(3, true),  // h (read)
                storage(4, false), // slot_count (rw)
                storage(5, false), // cursor (rw)
                storage(6, false), // cell_start (rw)
                storage(7, false), // sorted_idx (rw)
                storage(8, false), // dt_out (rw)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-sph-cfl-pl"),
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
        let pipeline_build = make_pipeline("build", "gpu-sph-cfl-build");
        let pipeline_cfl = make_pipeline("cfl_main", "gpu-sph-cfl-main");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-cfl-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuCfl {
            device,
            queue,
            bgl,
            pipeline_build,
            pipeline_cfl,
            params_buf,
            pos_buf: None,
            vel_buf: None,
            h_buf: None,
            slot_count_buf: None,
            cursor_buf: None,
            cell_start_buf: None,
            sorted_idx_buf: None,
            dt_out_buf: None,
            dt_readback: None,
            capacity_n: 0,
            table_size: 0,
        })
    }

    /// Per-target stable step `dt_i = C_cfl · h_i / v_sig,i`, gathering per target over
    /// the global `SUPPORT·h_max`. `h` is supplied (the density pass ran first); every
    /// slice must have length `pos.len()`. Empty input ⇒ empty output.
    pub fn per_target_dt(
        &mut self,
        pos: &[DVec3],
        vel: &[DVec3],
        h: &[f64],
        params: &HydroParams,
        c_cfl: f64,
    ) -> Vec<f64> {
        let n = pos.len();
        assert_eq!(vel.len(), n, "vel length mismatch");
        assert_eq!(h.len(), n, "h length mismatch");
        assert!(
            params.sound_speed() > 0.0,
            "GPU CFL needs a positive sound speed (keeps dt_i finite)"
        );
        if n == 0 {
            return Vec::new();
        }
        let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(
            h_max.is_finite() && h_max > 0.0,
            "GPU CFL needs positive finite smoothing lengths"
        );
        // Global gather radius = SUPPORT·h_max (load-bearing — see the module doc). The
        // build cell equals it, so the centered walk spans ≈2 cells and captures every
        // force-coupled neighbor of every target regardless of the h dynamic range.
        let radius = SUPPORT * h_max;
        let cell = radius.max(1e-12);

        let table_size = table_size_for(n);
        self.ensure_capacity(n, table_size);
        let gp = Params {
            n: n as u32,
            table_mask: table_size - 1,
            cell: cell as f32,
            radius: radius as f32,
            sound_speed: params.sound_speed() as f32,
            c_cfl: c_cfl as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        self.upload_inputs(&gp, pos, vel, h);
        self.dispatch(n)
    }

    /// The stable-dt bound `min_i dt_i`, or `f64::INFINITY` for no gas (empty input) —
    /// a `0` here would falsely report that every dt is too large. The host `min` is
    /// exact and order-independent, so it carries no numerics beyond `per_target_dt`.
    pub fn max_stable_dt(
        &mut self,
        pos: &[DVec3],
        vel: &[DVec3],
        h: &[f64],
        params: &HydroParams,
        c_cfl: f64,
    ) -> f64 {
        self.per_target_dt(pos, vel, h, params, c_cfl)
            .into_iter()
            .fold(f64::INFINITY, f64::min)
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
        self.pos_buf = Some(f32s(3 * n64, "gpu-sph-cfl-pos", dst));
        self.vel_buf = Some(f32s(3 * n64, "gpu-sph-cfl-vel", dst));
        self.h_buf = Some(f32s(n64, "gpu-sph-cfl-h", dst));
        self.slot_count_buf = Some(u32s(ts64, "gpu-sph-cfl-slot-count"));
        self.cursor_buf = Some(u32s(ts64, "gpu-sph-cfl-cursor"));
        self.cell_start_buf = Some(u32s(ts64 + 1, "gpu-sph-cfl-cell-start"));
        self.sorted_idx_buf = Some(u32s(n64, "gpu-sph-cfl-sorted-idx"));
        self.dt_out_buf = Some(f32s(
            n64,
            "gpu-sph-cfl-dt-out",
            wgpu::BufferUsages::COPY_SRC,
        ));
        self.dt_readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-sph-cfl-dt-readback"),
            size: n64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.capacity_n = n;
        self.table_size = table_size;
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        let e = "buffers ensured";
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-sph-cfl-bind-group"),
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
                    resource: self.h_buf.as_ref().expect(e).as_entire_binding(),
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
                    resource: self.dt_out_buf.as_ref().expect(e).as_entire_binding(),
                },
            ],
        })
    }

    /// Upload params + interleaved f32 positions/velocities + f32 smoothing lengths.
    fn upload_inputs(&self, params: &Params, pos: &[DVec3], vel: &[DVec3], h: &[f64]) {
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
        let h32: Vec<f32> = h.iter().map(|&x| x as f32).collect();
        write(&self.h_buf, &h32);
    }

    /// Build the hash (single invocation) then run the CFL pass (per target), reading
    /// back the `n` per-target stable steps as `f64`.
    fn dispatch(&self, n: usize) -> Vec<f64> {
        let bg = self.bind_group();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-cfl-build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sph-cfl-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_cfl);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(QUERY_WG as usize) as u32, 1, 1);
        }
        let bytes = n as u64 * std::mem::size_of::<f32>() as u64;
        enc.copy_buffer_to_buffer(
            self.dt_out_buf.as_ref().expect("ensured"),
            0,
            self.dt_readback.as_ref().expect("ensured"),
            0,
            bytes,
        );
        self.queue.submit([enc.finish()]);

        let readback = self.dt_readback.as_ref().expect("ensured");
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
        let out: Vec<f64> = flat[..n].iter().map(|&x| x as f64).collect();
        drop(data);
        readback.unmap();
        out
    }
}

/// Hash-table size for `n` bodies: the next power of two ≥ `2n`, floored at 64 (same
/// policy as `sph_grid` / `sph_density` / `sph_hydro`). Power-of-two so the slot
/// reduction is a mask.
fn table_size_for(n: usize) -> u32 {
    let target = (2 * n).max(64) as u32;
    target.next_power_of_two()
}
