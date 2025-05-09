//! Contraction Hierarchies is used to optimize shortest path queries on weighted graphs, conventionally road networks.
//! 
//! Contraction Hierarchies traditionally has two stages:
//! 1. Graph pre-processing (optimization stage):
//!     * Obtain a contraction heuristic which determines the next node to contract for any given graph. Some heuristics can be entirely precomputed without contracting the graph, in which case we can establish a contraction heuristic funtion by iterating through the precomputed list and ignoring the graph.
//!     * Successively contract the graph (with each intermediate graph at iteration `i` being called the `i`th core graph), adding shortcuts to preserve the shortest path distance between nodes which would otherwise be destroyed by the contraction.
//! 2. Path finding (query stage):
//!     * Obtain the eponymous "Contraction Hierarchy" by taking the union of all core graphs; in other words, the original graph augmented with all the shortcuts.
//!     * Perform bidirectional Dijkstra's algorithm on the contraction hierarchy to find the shortest path between any two nodes.
//! 
//! Our method performs only the optimization stage, using the `i`th core graph for `i` parameterized as a percent of the graph's cardinality, and does not perform any path finding.
//! This is because we are interested in reducing the size of the graph while preserving the nodes which are most likely to appear in most probable paths, but we are not concerned with always picking the most probable path. The Contraction Hierarchy (being the union of all core graphs) is actually larger than the initial graph, not smaller.

/// Core implementation of the Contraction Hierarchies algorithm
mod ch;
/// Heuristics for determining node contraction order
mod heuristic;
/// Utility functions for shortest path algorithms
mod utils;

pub use ch::{
    CHEdge,
    CHNode,
    Distance,
    CH,
};
pub use heuristic::{
    HeuristicGraph,
    nested_dissection_contraction_order,
};
