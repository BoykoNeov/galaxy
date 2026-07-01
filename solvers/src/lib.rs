//! `galaxy-solvers`: `ForceSolver` implementations.
//!
//! `DirectSum` is the exact O(N²) oracle and the small-N workhorse;
//! `BarnesHut` is the O(N log N) tree approximation validated against it.
//! PM / TreePM solvers join here later behind the same trait.

pub mod barnes_hut;
pub mod direct_sum;
pub mod lbvh;
pub mod potential;

pub use barnes_hut::{BarnesHut, BuildMode, FlatNode, FlatTree};
pub use direct_sum::DirectSum;
pub use lbvh::{
    reference_aggregate, reference_flatten, reference_karras, reference_morton, reference_sort,
    KarrasAgg, KarrasTree, Lbvh, LbvhFlat, LbvhNode, MortonBounds, MortonReference, NO_PARENT,
};
