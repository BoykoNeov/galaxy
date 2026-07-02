//! [`GpuResidentLeapfrog`]: GPU-**resident** leapfrog stepping (DESIGN M4i) — keeping particle
//! *state* on the GPU across integrator steps, the payoff M4h's single-device fuse unlocked.
//!
//! ## What residency buys, and what it costs
//! [`crate::GpuLbvhFused`] (M4h) fused the whole LBVH build+traverse onto one device in one
//! submit, but still **uploads state and reads back accel every `accelerations` call** — one
//! CPU↔GPU round-trip per force evaluation. `GpuResidentLeapfrog` closes that loop: `pos`, `vel`,
//! `mass`, and `acc` live in GPU storage buffers *across* steps, the kick/drift arithmetic runs
//! on the device, and nothing crosses the bus until an explicit [`snapshot`](Self::snapshot).
//! `bodies` (xyz=pos, w=mass) — already the traversal's input — doubles as the resident position
//! buffer, so the force pipeline reads state in place.
//!
//! This is *not* a throughput speedup (the M4h serial stages — sort, aggregate, flatten — are
//! unchanged and still dominate); it removes the per-step sync points, the point of residency.
//! Each [`step`](Self::step) is still one submit; **batching K steps into a single encoder** (to
//! also drop per-submit overhead) is the named follow-up — residency and batching are distinct,
//! and one-submit-per-step is already fully resident.
//!
//! ## The precision cost is real and documented
//! The host-driven path ([`galaxy_core::LeapfrogKdk`] + a solver) keeps **authoritative
//! positions in f64** and re-narrows to f32 only to feed the GPU force kernel each step. The
//! resident path accumulates `pos += vel*dt` **in f32 across every step**, because WGSL has no
//! portable `f64`. Position updates below ~1e-6 of a coordinate's magnitude are lost, so energy
//! drifts more than the f64 leapfrog's clean bounded oscillation. This is acceptable for the
//! render money-shot (the renderer is f32 anyway) and mirrors the existing f32-force / f64-energy
//! inconsistency; **double-single (float-float) position accumulation is the deferred precision
//! follow-up.**
//!
//! ## Not a `ForceSolver`
//! The [`galaxy_core::ForceSolver`] interface is host-state-in / accel-out — fundamentally
//! incompatible with keeping state resident. So this is its own type with an
//! `upload → step* → snapshot` lifecycle, exactly as DESIGN "Remaining M4+" anticipated.

use bytemuck::{Pod, Zeroable};

use galaxy_core::{DVec3, State};

use crate::fused_core::{bg_entry, storage_entry, uniform_entry, FusedCore};
use crate::GpuError;

/// Workgroup width for the per-particle kick/drift/reset kernels.
const WG: u32 = 256;

/// `NO_PARENT` = `u32::MAX`, the root sentinel the aggregate walk stops on. The reset kernel
/// re-writes every `parent` slot to this each force evaluation (see [`galaxy_solvers::NO_PARENT`]).
const NO_PARENT_LIT: &str = "4294967295u";

/// Re-seed `idx_a` (iota) and `parent` (`NO_PARENT`) on the GPU each force evaluation — the
/// on-device equivalent of the fused solver's per-call host `write_buffer`s, so no state leaves
/// the device between steps. `parent` needs the `NO_PARENT` pre-fill because the Karras build only
/// writes children's parent slots (never the root's), and the aggregate walk stops on it.
fn reset_shader() -> String {
    format!(
        r#"
struct Params {{ n: u32, dt: f32, half_dt: f32, pad: u32 }};
@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> idx_a:  array<u32>;
@group(0) @binding(2) var<storage, read_write> parent: array<u32>;

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let k = gid.x;
    let total = 2u * params.n - 1u;
    if (k >= total) {{ return; }}
    if (k < params.n) {{ idx_a[k] = k; }}
    parent[k] = {NO_PARENT_LIT};
}}
"#
    )
}

/// Leapfrog half-kick: `vel.xyz += accel.xyz * half_dt`, preserving `vel.w`.
const KICK_SHADER: &str = r#"
struct Params { n: u32, dt: f32, half_dt: f32, pad: u32 };
@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> vel:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       accel:  array<vec4<f32>>;

@compute @workgroup_size(256)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let v = vel[i];
    vel[i] = vec4<f32>(v.xyz + accel[i].xyz * params.half_dt, v.w);
}
"#;

/// Leapfrog drift: `pos.xyz += vel.xyz * dt`, preserving `bodies.w` (= mass). `bodies` is the
/// traversal's own position buffer, so this advances the state the force pipeline reads next.
const DRIFT_SHADER: &str = r#"
struct Params { n: u32, dt: f32, half_dt: f32, pad: u32 };
@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> bodies: array<vec4<f32>>; // xyz=pos, w=mass
@group(0) @binding(2) var<storage, read>       vel:    array<vec4<f32>>;

@compute @workgroup_size(256)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = bodies[i];
    bodies[i] = vec4<f32>(b.xyz + vel[i].xyz * params.dt, b.w);
}
"#;

/// Uniform for the kick/drift/reset kernels: particle count + this step's `dt` / `half_dt`.
/// `reset` reads only `.n`; `kick` reads `.half_dt`; `drift` reads `.dt`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepParams {
    n: u32,
    dt: f32,
    half_dt: f32,
    _pad: u32,
}

/// Resident-owned resources (the resident velocity buffer + the readback buffers + the
/// kick/drift/reset bind groups). Rebuilt when [`FusedCore`] grows.
struct ResidentResources {
    vel: wgpu::Buffer,
    pos_readback: wgpu::Buffer,
    vel_readback: wgpu::Buffer,
    reset_bg: wgpu::BindGroup,
    kick_bg: wgpu::BindGroup,
    drift_bg: wgpu::BindGroup,
    capacity: usize,
}

/// GPU-resident kick-drift-kick leapfrog over the M4h fused LBVH force pipeline. State stays in
/// GPU buffers across [`step`](Self::step)s; only [`snapshot`](Self::snapshot) reads it back.
pub struct GpuResidentLeapfrog {
    core: FusedCore,
    // kick/drift/reset pipelines + layouts (built once).
    reset_pl: wgpu::ComputePipeline,
    kick_pl: wgpu::ComputePipeline,
    drift_pl: wgpu::ComputePipeline,
    reset_bgl: wgpu::BindGroupLayout,
    kick_bgl: wgpu::BindGroupLayout,
    drift_bgl: wgpu::BindGroupLayout,
    step_params_buf: wgpu::Buffer,
    res: Option<ResidentResources>,
    // Host-tracked bookkeeping.
    n: usize,
    time: f64,
    mass: Vec<f64>,
}

impl GpuResidentLeapfrog {
    /// Bring up the resident compute device + every pipeline (the shared [`FusedCore`] build/
    /// traverse plus the new kick/drift/reset kernels). Returns a typed [`GpuError`] on adapter/
    /// device failure.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        let core = FusedCore::new(g, softening, theta)?;
        let dev = &core.device;

        let module = |label: &str, src: &str| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let reset_mod = module("resident-reset", &reset_shader());
        let kick_mod = module("resident-kick", KICK_SHADER);
        let drift_mod = module("resident-drift", DRIFT_SHADER);

        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        // reset: 0 uniform, 1 idx_a(rw), 2 parent(rw)
        let reset_bgl = bgl(
            "resident-reset-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        );
        // kick: 0 uniform, 1 vel(rw), 2 accel(r)
        let kick_bgl = bgl(
            "resident-kick-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        );
        // drift: 0 uniform, 1 bodies(rw), 2 vel(r)
        let drift_bgl = bgl(
            "resident-drift-bgl",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        );

        let pipeline = |label: &str,
                        layout: &wgpu::BindGroupLayout,
                        module: &wgpu::ShaderModule,
                        entry: &str| {
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let reset_pl = pipeline("resident-reset", &reset_bgl, &reset_mod, "reset");
        let kick_pl = pipeline("resident-kick", &kick_bgl, &kick_mod, "kick");
        let drift_pl = pipeline("resident-drift", &drift_bgl, &drift_mod, "drift");

        let step_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident-step-params"),
            size: std::mem::size_of::<StepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuResidentLeapfrog {
            core,
            reset_pl,
            kick_pl,
            drift_pl,
            reset_bgl,
            kick_bgl,
            drift_bgl,
            step_params_buf,
            res: None,
            n: 0,
            time: 0.0,
            mass: Vec::new(),
        })
    }

    /// (Re)allocate the resident velocity + readback buffers and rebuild the kick/drift/reset bind
    /// groups (which reference [`FusedCore`]'s `bodies`/`accel`/`idx_a`/`parent`). Called after
    /// `core.ensure_capacity`, so it sees the current core buffers. `cap >= 2` — allocated even for
    /// a single particle so no intermediate buffer is zero-sized.
    fn ensure_capacity(&mut self, cap: usize) {
        if let Some(res) = &self.res {
            if cap <= res.capacity {
                return;
            }
        }
        let dev = &self.core.device;
        let f4 = |count: usize| (count * std::mem::size_of::<[f32; 4]>()) as u64;
        let store = wgpu::BufferUsages::STORAGE;
        let cdst = wgpu::BufferUsages::COPY_DST;
        let csrc = wgpu::BufferUsages::COPY_SRC;
        let mapread = wgpu::BufferUsages::MAP_READ;

        let vel = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident-vel"),
            size: f4(cap),
            usage: store | cdst | csrc,
            mapped_at_creation: false,
        });
        let make_rb = |label: &str| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: f4(cap),
                usage: cdst | mapread,
                mapped_at_creation: false,
            })
        };
        let pos_readback = make_rb("resident-pos-readback");
        let vel_readback = make_rb("resident-vel-readback");

        let core_res = self.core.res.as_ref().expect("core capacity ensured first");
        let bind =
            |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
                dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries,
                })
            };
        let reset_bg = bind(
            "resident-reset-bg",
            &self.reset_bgl,
            &[
                bg_entry(0, &self.step_params_buf),
                bg_entry(1, &core_res.idx_a),
                bg_entry(2, &core_res.parent),
            ],
        );
        let kick_bg = bind(
            "resident-kick-bg",
            &self.kick_bgl,
            &[
                bg_entry(0, &self.step_params_buf),
                bg_entry(1, &vel),
                bg_entry(2, &core_res.accel),
            ],
        );
        let drift_bg = bind(
            "resident-drift-bg",
            &self.drift_bgl,
            &[
                bg_entry(0, &self.step_params_buf),
                bg_entry(1, &core_res.bodies),
                bg_entry(2, &vel),
            ],
        );

        self.res = Some(ResidentResources {
            vel,
            pos_readback,
            vel_readback,
            reset_bg,
            kick_bg,
            drift_bg,
            capacity: cap,
        });
    }

    /// Write the per-step uniform (`n`, `dt`, `half_dt`). `dt = 0` for the initial prime (only
    /// `.n` is read there).
    fn write_step_params(&self, dt: f64) {
        self.core.queue.write_buffer(
            &self.step_params_buf,
            0,
            bytemuck::bytes_of(&StepParams {
                n: self.n as u32,
                dt: dt as f32,
                half_dt: (0.5 * dt) as f32,
                _pad: 0,
            }),
        );
    }

    /// A one-invocation-per-particle compute pass over the given pipeline + bind group.
    fn per_particle_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
    ) {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bg, &[]);
        p.dispatch_workgroups((self.n as u32).div_ceil(WG), 1, 1);
    }

    /// Append the force evaluation into `accel`: for `n >= 2` the reset (re-seed idx_a/parent)
    /// followed by the shared build+traverse; for `n == 1` just zero `accel` (a lone particle
    /// feels no force). Assumes `bodies` holds the current positions.
    fn encode_force(&self, enc: &mut wgpu::CommandEncoder) {
        let res = self.res.as_ref().expect("resident resources ensured");
        if self.n >= 2 {
            self.per_particle_pass(enc, "resident-reset", &self.reset_pl, &res.reset_bg);
            self.core.encode_build_traverse(enc, self.n);
        } else {
            // n == 1: no tree; the single particle's acceleration is exactly zero.
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            enc.clear_buffer(&core_res.accel, 0, None);
        }
    }

    /// Upload `state` (f64→f32 narrowed) into the resident GPU buffers, (re)allocating as `N`
    /// changes, and **prime** the acceleration (one force evaluation, no readback) so the first
    /// [`step`](Self::step)'s opening half-kick uses `a(x₀)`, not a stale value. Resets the clock.
    pub fn upload(&mut self, state: &State) {
        let n = state.len();
        self.n = n;
        self.time = 0.0;
        self.mass = state.mass.clone();
        if n == 0 {
            return;
        }

        let cap = n.max(2); // never size intermediate buffers to zero
        self.core.ensure_capacity(cap);
        self.ensure_capacity(cap);
        self.core.write_uniforms(n);
        self.write_step_params(0.0);

        // Upload positions (bodies: xyz=pos, w=mass) and velocities.
        let bodies: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let p = state.pos[i];
                [p.x as f32, p.y as f32, p.z as f32, state.mass[i] as f32]
            })
            .collect();
        let vels: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let v = state.vel[i];
                [v.x as f32, v.y as f32, v.z as f32, 0.0]
            })
            .collect();
        {
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            let res = self.res.as_ref().expect("resident resources ensured");
            self.core
                .queue
                .write_buffer(&core_res.bodies, 0, bytemuck::cast_slice(&bodies));
            self.core
                .queue
                .write_buffer(&res.vel, 0, bytemuck::cast_slice(&vels));
        }

        // Prime accel = a(x₀). No readback.
        let mut enc = self
            .core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident-prime-encoder"),
            });
        self.encode_force(&mut enc);
        self.core.queue.submit([enc.finish()]);
    }

    /// Advance one resident KDK step by `dt`: kick½ · drift · (reset+build+traverse into `acc`) ·
    /// kick½ — one submit, no readback. Requires a prior [`upload`](Self::upload).
    pub fn step(&mut self, dt: f64) {
        if self.n == 0 {
            self.time += dt;
            return;
        }
        self.write_step_params(dt);
        let res = self.res.as_ref().expect("resident resources ensured");

        let mut enc = self
            .core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident-step-encoder"),
            });
        // Kick½ with a(xₙ) [carried from the previous step's closing kick, or the prime].
        self.per_particle_pass(&mut enc, "resident-kick-open", &self.kick_pl, &res.kick_bg);
        // Drift: xₙ → xₙ₊₁.
        self.per_particle_pass(&mut enc, "resident-drift", &self.drift_pl, &res.drift_bg);
        // Recompute a(xₙ₊₁) in place (accel), left resident for the next step's opening kick.
        self.encode_force(&mut enc);
        // Kick½ with a(xₙ₊₁).
        self.per_particle_pass(&mut enc, "resident-kick-close", &self.kick_pl, &res.kick_bg);
        self.core.queue.submit([enc.finish()]);

        self.time += dt;
    }

    /// Advance `steps` resident KDK steps of `dt` (each its own submit; still fully resident).
    pub fn step_many(&mut self, dt: f64, steps: u64) {
        for _ in 0..steps {
            self.step(dt);
        }
    }

    /// Read the resident state back to the host as a fresh [`State`] (pos/vel widened f32→f64,
    /// mass/time host-tracked). The only device→host transfer.
    pub fn snapshot(&mut self) -> State {
        let n = self.n;
        if n == 0 {
            let mut s = State::from_phase_space(vec![], vec![], vec![]);
            s.time = self.time;
            return s;
        }

        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        let (pos, vel) = {
            let core_res = self.core.res.as_ref().expect("core resources ensured");
            let res = self.res.as_ref().expect("resident resources ensured");

            let mut enc =
                self.core
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("resident-snapshot-encoder"),
                    });
            enc.copy_buffer_to_buffer(&core_res.bodies, 0, &res.pos_readback, 0, bytes);
            enc.copy_buffer_to_buffer(&res.vel, 0, &res.vel_readback, 0, bytes);
            self.core.queue.submit([enc.finish()]);

            let pos = self.read_vec3(&res.pos_readback, bytes);
            let vel = self.read_vec3(&res.vel_readback, bytes);
            (pos, vel)
        };

        let mut s = State::from_phase_space(pos, vel, self.mass.clone());
        s.time = self.time;
        s
    }

    /// Map a f32 `vec4` buffer, block once, and widen the xyz lanes of the first `n` entries to
    /// f64. A map failure is an exceptional GPU loss and panics rather than return corrupt state.
    fn read_vec3(&self, readback: &wgpu::Buffer, bytes: u64) -> Vec<DVec3> {
        let slice = readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        rx.recv()
            .expect("map channel closed")
            .expect("gpu buffer map failed");

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let out = (0..self.n)
            .map(|i| {
                let b = i * 4;
                DVec3::new(floats[b] as f64, floats[b + 1] as f64, floats[b + 2] as f64)
            })
            .collect();
        drop(data);
        readback.unmap();
        out
    }

    /// Simulation time after the steps taken so far.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Number of resident particles (0 before the first [`upload`](Self::upload)).
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether no particles are resident.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}
