#![deny(
    // This is overly strict, of course. The intent is somewhat of a "quality seal," less to fix everything, and more to force us to add inline allows, which are even more needlessly verbose, but give us a mechanism to say "we think this is okay, but you might want to take a second look here."
    clippy::nursery,
    clippy::pedantic,
    // These are also just for clinic purposes
    missing_docs,
    clippy::missing_docs_in_private_items,
)]

//! # sk-gen – Synthetic Kubernetes trace generator for `SimKube`
//!
//! sk-gen turns one or more real `SimKube` traces into a probabilistic state-transition graph,
//! expands that graph with hypothetical actions, contracts it using Contraction Hierarchies, and
//! finally samples random walks to produce new, realistic-looking traces.
//!
// TODO: this only works with --document-private-items so as to satifsfy clinic requirements, we may want to remove these referenes to make more user facing.
//! ## Pipeline overview
//! 1. Graph construction ([`construct_graph`](crate::simulation::construct_graph)) – Convert
//!    observed `TraceEvent`s in an `ExportedTrace` into a [`petgraph::Graph`] whose nodes are
//!    Kubernetes states (`Node`) and whose edges represent state–transition actions (`Edge`).
//! 2. Graph expansion ([`expand_graph`](crate::simulation::expand_graph)) – Breadth-first
//!    application of a user-supplied `next_action_fn` that yields "atomic" changes (for example
//!    ±CPU, ±memory, ±replica count). Probabilities extracted from the input trace are attached to
//!    these edges and normalised so that the outgoing weights of each node sum to 1.
//! 3. Graph contraction ([`contract_graph`](crate::simulation::contract_graph)) – Apply a Nested
//!    Dissection contraction ordering to remove low-centrality nodes while inserting shortcut edges
//!    that merge sequences of atomic actions into single "composite" actions with combined
//!    probabilities.
//! 4. Trace generation ([`generate_traces`](crate::simulation::generate_traces)) – Perform weighted
//!    random walks ([`walks_with_sampling`](crate::simulation::walks_with_sampling)) through the
//!    contracted graph and write each walk out as a JSON file compatible with `SimKube`'s
//!    `TraceStore`.
//!
//! The entry point [`simulation::run`] orchestrates these stages and dumps
//! intermediate artefacts (DOT files, synthetic traces, run metadata) into
//! a timestamped directory under `runs/`.
//!
//! All long-running stages are annotated with [`tracing`]-powered spans so
//! that callers can observe progress and timing.


pub mod contraction_hierarchies;
pub mod model;
pub mod simulation;
pub mod utils;

pub use model::{
    Action,
    DynamicObjectNewType,
    Node,
    ObjectKey,
};
pub use utils::diff_objects;
