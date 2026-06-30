use galaxy_core::{DVec3, ForceSolver, State};

/// Barnes-Hut octree solver (monopole-only). An O(N log N) approximation to
/// direct summation, controlled by the opening angle `theta`: a node is used as
/// a single softened point mass when `node_size / distance < theta`, else its
/// children are opened. As `theta -> 0` it reproduces direct summation (same
/// Plummer-softened kernel) to roundoff. Quadrupole terms are a later refinement.
#[derive(Clone, Copy, Debug)]
pub struct BarnesHut {
    /// Gravitational constant.
    pub g: f64,
    /// Plummer softening length ε (use the same value as the oracle to compare).
    pub softening: f64,
    /// Opening angle θ. Smaller = more accurate, more work.
    pub theta: f64,
}

impl BarnesHut {
    pub fn new(g: f64, softening: f64, theta: f64) -> Self {
        Self {
            g,
            softening,
            theta,
        }
    }
}

const SENTINEL: u32 = u32::MAX;
const LEAF_CAP: usize = 1;
/// Stop subdividing once a cell is this fraction of the root size; coincident or
/// near-coincident particles then share a bucket leaf (resolved by direct sum).
const MIN_HALF_FRAC: f64 = 1e-10;

struct Node {
    center: DVec3,
    half: f64,
    mass: f64,
    com: DVec3,
    /// Distance from the geometric center to the center of mass. The COM can sit
    /// near a cell edge ("detached"), so the nearest mass may be `delta` closer to
    /// a target than the COM is; the opening criterion subtracts it (Barnes 1994).
    delta: f64,
    children: [u32; 8],
    bodies: Vec<u32>,
    leaf: bool,
}

impl Node {
    fn leaf(center: DVec3, half: f64) -> Self {
        Node {
            center,
            half,
            mass: 0.0,
            com: DVec3::ZERO,
            delta: 0.0,
            children: [SENTINEL; 8],
            bodies: Vec::new(),
            leaf: true,
        }
    }
}

fn octant(center: DVec3, p: DVec3) -> usize {
    let mut o = 0;
    if p.x >= center.x {
        o |= 1;
    }
    if p.y >= center.y {
        o |= 2;
    }
    if p.z >= center.z {
        o |= 4;
    }
    o
}

fn child_center(center: DVec3, half: f64, oct: usize) -> DVec3 {
    let q = 0.5 * half;
    DVec3::new(
        center.x + if oct & 1 != 0 { q } else { -q },
        center.y + if oct & 2 != 0 { q } else { -q },
        center.z + if oct & 4 != 0 { q } else { -q },
    )
}

struct Octree {
    nodes: Vec<Node>,
    min_half: f64,
}

impl Octree {
    fn build(pos: &[DVec3], mass: &[f64]) -> Octree {
        let mut lo = pos[0];
        let mut hi = pos[0];
        for &p in pos {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let center = (lo + hi) * 0.5;
        // Cube half-side that strictly contains every particle (slight padding).
        let half = (0.5 * (hi - lo).max_element()).max(1e-12) * (1.0 + 1e-9);
        let mut tree = Octree {
            nodes: vec![Node::leaf(center, half)],
            min_half: half * MIN_HALF_FRAC,
        };
        for (i, _) in pos.iter().enumerate() {
            tree.insert(0, i as u32, pos);
        }
        tree.aggregate(0, pos, mass);
        tree
    }

    fn insert(&mut self, node: usize, b: u32, pos: &[DVec3]) {
        if self.nodes[node].leaf {
            self.nodes[node].bodies.push(b);
            if self.nodes[node].bodies.len() > LEAF_CAP && self.nodes[node].half > self.min_half {
                self.subdivide(node, pos);
            }
        } else {
            let oct = octant(self.nodes[node].center, pos[b as usize]);
            let child = self.ensure_child(node, oct);
            self.insert(child, b, pos);
        }
    }

    fn subdivide(&mut self, node: usize, pos: &[DVec3]) {
        let bodies = std::mem::take(&mut self.nodes[node].bodies);
        self.nodes[node].leaf = false;
        for b in bodies {
            let oct = octant(self.nodes[node].center, pos[b as usize]);
            let child = self.ensure_child(node, oct);
            self.insert(child, b, pos);
        }
    }

    fn ensure_child(&mut self, node: usize, oct: usize) -> usize {
        let existing = self.nodes[node].children[oct];
        if existing != SENTINEL {
            return existing as usize;
        }
        let center = child_center(self.nodes[node].center, self.nodes[node].half, oct);
        let half = 0.5 * self.nodes[node].half;
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node::leaf(center, half));
        self.nodes[node].children[oct] = idx;
        idx as usize
    }

    fn aggregate(&mut self, node: usize, pos: &[DVec3], mass: &[f64]) {
        let mut m = 0.0;
        let mut c = DVec3::ZERO;
        if self.nodes[node].leaf {
            for &b in &self.nodes[node].bodies {
                m += mass[b as usize];
                c += pos[b as usize] * mass[b as usize];
            }
        } else {
            let children = self.nodes[node].children;
            for &ch in &children {
                if ch != SENTINEL {
                    self.aggregate(ch as usize, pos, mass);
                    m += self.nodes[ch as usize].mass;
                    c += self.nodes[ch as usize].com * self.nodes[ch as usize].mass;
                }
            }
        }
        self.nodes[node].mass = m;
        let com = if m > 0.0 {
            c / m
        } else {
            self.nodes[node].center
        };
        self.nodes[node].com = com;
        self.nodes[node].delta = (com - self.nodes[node].center).length();
    }

    fn accel_node(&self, node: usize, target: usize, q: &Query) -> DVec3 {
        let nd = &self.nodes[node];
        if nd.mass <= 0.0 {
            return DVec3::ZERO;
        }
        let xp = q.pos[target];
        if nd.leaf {
            let mut a = DVec3::ZERO;
            for &b in &nd.bodies {
                if b as usize == target {
                    continue;
                }
                let dx = q.pos[b as usize] - xp;
                let r2 = dx.length_squared() + q.eps2;
                a += dx * (q.mass[b as usize] / (r2 * r2.sqrt()));
            }
            return a;
        }
        // Never approximate a cell that contains the target as a single mass.
        let inside = (xp.x - nd.center.x).abs() <= nd.half
            && (xp.y - nd.center.y).abs() <= nd.half
            && (xp.z - nd.center.z).abs() <= nd.half;
        let dx = nd.com - xp;
        let d2 = dx.length_squared();
        let s = 2.0 * nd.half;
        // Barnes (1994) criterion: accept the multipole when the cell subtends a
        // small angle from the *nearest mass it could hold*, not just from the COM.
        // The nearest mass may be `delta` closer than the COM, so require
        //   s / (d − delta) ≤ θ   ⟺   s ≤ θ·(d − delta),
        // which reduces to the classic s/d ≤ θ for a centered COM (delta = 0) and
        // forces opening when a detached COM would otherwise hide a near particle.
        let d = d2.sqrt();
        if !inside && q.theta * (d - nd.delta) >= s {
            let r2 = d2 + q.eps2;
            return dx * (nd.mass / (r2 * r2.sqrt()));
        }
        let mut a = DVec3::ZERO;
        for &ch in &nd.children {
            if ch != SENTINEL {
                a += self.accel_node(ch as usize, target, q);
            }
        }
        a
    }
}

/// Read-only context threaded through the force traversal.
struct Query<'a> {
    pos: &'a [DVec3],
    mass: &'a [f64],
    theta: f64,
    eps2: f64,
}

impl ForceSolver for BarnesHut {
    #[allow(clippy::needless_range_loop)]
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        let tree = Octree::build(&state.pos, &state.mass);
        let q = Query {
            pos: &state.pos,
            mass: &state.mass,
            theta: self.theta,
            eps2: self.softening * self.softening,
        };
        for i in 0..n {
            acc[i] = tree.accel_node(0, i, &q) * self.g;
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn potential_energy(&self, state: &State) -> f64 {
        // Exact softened potential (O(N^2)); a global diagnostic, not yet
        // tree-accelerated. Matches DirectSum's potential exactly.
        let n = state.len();
        let eps2 = self.softening * self.softening;
        let mut u = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = state.pos[j] - state.pos[i];
                let r = (dx.length_squared() + eps2).sqrt();
                u -= self.g * state.mass[i] * state.mass[j] / r;
            }
        }
        u
    }
}
