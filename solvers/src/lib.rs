//! `galaxy-solvers`: `ForceSolver` implementations.
//!
//! `DirectSum` is the exact O(N²) oracle and the small-N workhorse;
//! `BarnesHut` is the O(N log N) tree approximation validated against it.
//! PM / TreePM solvers join here later behind the same trait.

pub mod barnes_hut;
pub mod direct_sum;

pub use barnes_hut::BarnesHut;
pub use direct_sum::DirectSum;
