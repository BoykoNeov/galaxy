//! Export a two-galaxy collision for the REBOUND cross-check harness.
//!
//! Builds a fixed parabolic encounter, dumps the initial conditions and our
//! engine's diagnostics as NumPy `.npy` arrays (plus a `params.json`), so the
//! committed Python harness (`validate/rebound/cross_check.py`) can integrate the
//! SAME initial conditions with REBOUND IAS15 and cross-check.
//!
//! Run (release strongly recommended — DirectSum is O(N^2)):
//!   cargo run -p galaxy-validate --release --example export_collision -- <outdir>
//!
//! Output layout:
//!   <outdir>/ic/{pos,vel,mass,progenitor,acc}.npy   initial conditions + IC forces
//!   <outdir>/ours/{time,energy,rh0,rh1}.npy          our run's diagnostics vs time
//!   <outdir>/ours/{final_pos,final_vel}.npy          our final state
//!   <outdir>/params.json                             G, softening, dt, counts, ...

use std::path::{Path, PathBuf};

use galaxy_core::{
    diagnostics, DVec3, ForceSolver, LeapfrogKdk, Progenitor, State, StaticBackground,
};
use galaxy_ic::{Collision, Plummer};
use galaxy_io::Header;
use galaxy_sim::{run, SimConfig, SimError, SnapshotSink};
use galaxy_solvers::DirectSum;
use galaxy_validate::npy;

// Fixed scenario — kept in sync with params.json (written below) so the harness
// reconstructs the identical system.
const G: f64 = 1.0;
const M1: f64 = 1.0;
const A1: f64 = 1.0;
const M2: f64 = 0.6;
const A2: f64 = 0.8;
const ECC: f64 = 1.0; // parabolic (Toomre tidal-tail encounter)
const PERI: f64 = 1.5;
const SEP: f64 = 8.0;
const EPS: f64 = 0.05;
const N1: usize = 300;
const N2: usize = 200;
const SEED: u64 = 0xDEAD_BEEF;
const DT: f64 = 0.02;
const N_STEPS: u64 = 1000;
const SNAPSHOT_EVERY: u64 = 25;

/// Median particle radius of one progenitor about its own mass-weighted COM.
/// With equal-mass particles within a galaxy this is its half-mass radius — the
/// statistical quantity the harness compares between the two integrators.
fn half_mass_radius(state: &State, prog: Progenitor) -> f64 {
    let mut com = DVec3::ZERO;
    let mut m_tot = 0.0;
    for i in 0..state.len() {
        if state.progenitor[i] == prog {
            com += state.pos[i] * state.mass[i];
            m_tot += state.mass[i];
        }
    }
    com /= m_tot;
    let mut radii: Vec<f64> = (0..state.len())
        .filter(|&i| state.progenitor[i] == prog)
        .map(|i| (state.pos[i] - com).length())
        .collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    radii[radii.len() / 2]
}

/// Sink that records the diagnostics we cross-check, plus the final state.
struct ExportSink {
    solver: DirectSum,
    time: Vec<f64>,
    energy: Vec<f64>,
    rh0: Vec<f64>,
    rh1: Vec<f64>,
    final_state: Option<State>,
}

impl SnapshotSink for ExportSink {
    fn emit(&mut self, _header: &Header, state: &State) -> Result<(), SimError> {
        self.time.push(state.time);
        self.energy
            .push(diagnostics::total_energy(state, &self.solver));
        self.rh0.push(half_mass_radius(state, Progenitor(0)));
        self.rh1.push(half_mass_radius(state, Progenitor(1)));
        self.final_state = Some(state.clone());
        Ok(())
    }
}

fn main() {
    let outdir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("galaxy_collision_export"));

    let collision = Collision::new(
        Plummer::new(G, M1, A1),
        Plummer::new(G, M2, A2),
        ECC,
        PERI,
        SEP,
    );
    let ic = collision.sample(N1, N2, SEED);
    let n = ic.len();

    // IC forces under the exact softened oracle — the harness compares these to
    // REBOUND's t=0 accelerations (the unambiguous "same physical system" check).
    let mut solver = DirectSum::new(G, EPS);
    let mut acc = vec![DVec3::ZERO; n];
    solver.accelerations(&ic, &mut acc);
    let ic_energy = diagnostics::total_energy(&ic, &solver);

    let ic_dir = outdir.join("ic");
    let ours_dir = outdir.join("ours");
    std::fs::create_dir_all(&ic_dir).expect("create ic dir");
    std::fs::create_dir_all(&ours_dir).expect("create ours dir");

    npy::write_vec3_file(ic_dir.join("pos.npy"), &ic.pos).unwrap();
    npy::write_vec3_file(ic_dir.join("vel.npy"), &ic.vel).unwrap();
    npy::write_f64_file(ic_dir.join("mass.npy"), &ic.mass).unwrap();
    npy::write_vec3_file(ic_dir.join("acc.npy"), &acc).unwrap();
    let progenitor: Vec<f64> = ic.progenitor.iter().map(|p| p.0 as f64).collect();
    npy::write_f64_file(ic_dir.join("progenitor.npy"), &progenitor).unwrap();

    // Run our engine on this IC and record diagnostics at each snapshot.
    let mut state = ic.clone();
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = SimConfig {
        dt: DT,
        n_steps: N_STEPS,
        snapshot_every: SNAPSHOT_EVERY,
        softening: EPS,
        rng_seed: SEED,
        config_hash: 0,
        units: "nbody-G1".to_string(),
        sf: None,
    };
    let mut sink = ExportSink {
        solver: DirectSum::new(G, EPS),
        time: Vec::new(),
        energy: Vec::new(),
        rh0: Vec::new(),
        rh1: Vec::new(),
        final_state: None,
    };
    let mut run_solver = DirectSum::new(G, EPS);
    run(
        &mut state,
        &mut run_solver,
        &mut integ,
        &bg,
        &cfg,
        &mut sink,
    )
    .expect("run");

    npy::write_f64_file(ours_dir.join("time.npy"), &sink.time).unwrap();
    npy::write_f64_file(ours_dir.join("energy.npy"), &sink.energy).unwrap();
    npy::write_f64_file(ours_dir.join("rh0.npy"), &sink.rh0).unwrap();
    npy::write_f64_file(ours_dir.join("rh1.npy"), &sink.rh1).unwrap();
    let final_state = sink.final_state.expect("at least one snapshot");
    npy::write_vec3_file(ours_dir.join("final_pos.npy"), &final_state.pos).unwrap();
    npy::write_vec3_file(ours_dir.join("final_vel.npy"), &final_state.vel).unwrap();

    write_params(&outdir, n, ic_energy);

    eprintln!("Exported collision to {}", outdir.display());
    eprintln!("Cross-check with REBOUND:");
    eprintln!(
        "  python validate/rebound/cross_check.py {}",
        outdir.display()
    );
}

fn write_params(outdir: &Path, n: usize, ic_energy: f64) {
    // Hand-written JSON (no serde dep). {:?} on f64 round-trips exactly.
    let json = format!(
        "{{\n  \"g\": {g:?},\n  \"softening\": {eps:?},\n  \"dt\": {dt:?},\n  \"n_steps\": {steps},\n  \"snapshot_every\": {every},\n  \"n_particles\": {n},\n  \"n1\": {n1},\n  \"n2\": {n2},\n  \"seed\": {seed},\n  \"eccentricity\": {ecc:?},\n  \"pericenter\": {peri:?},\n  \"separation\": {sep:?},\n  \"ic_softened_energy\": {ic_energy:?}\n}}\n",
        g = G,
        eps = EPS,
        dt = DT,
        steps = N_STEPS,
        every = SNAPSHOT_EVERY,
        n = n,
        n1 = N1,
        n2 = N2,
        seed = SEED,
        ecc = ECC,
        peri = PERI,
        sep = SEP,
        ic_energy = ic_energy,
    );
    std::fs::write(outdir.join("params.json"), json).expect("write params.json");
}
