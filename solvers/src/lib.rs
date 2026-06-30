//! `galaxy-solvers`: `ForceSolver` implementations.
//!
//! `DirectSum` is the exact O(N²) oracle and the small-N workhorse. Tree /
//! PM / TreePM solvers join here later behind the same trait.

pub mod direct_sum;

pub use direct_sum::DirectSum;
