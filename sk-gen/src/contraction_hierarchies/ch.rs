use std::hash::Hash;

use anyhow::{
    Context,
    Result,
};
use itertools::Itertools;
use ordered_float::{
    Float,
    OrderedFloat,
};
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeIndex;
use petgraph::Graph;
use tracing::{debug, error, info};

use super::utils::dijkstra;

/// A wrapper on Edge which lets us mark a node as contracted during a particular iteration.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CHNode<Node> {
    /// An original node from the input graph.
    Original {
        /// The original node data.
        node: Node,
    },
    /// A node that has been contracted during the hierarchy construction.
    Contracted {
        /// The original node data.
        node: Node,
        /// The iteration number when the node was contracted.
        iteration: usize,
    },
}

impl<Node> CHNode<Node> {
    /// Creates a new `CHNode` in the Original state.
    const fn new_original(node: Node) -> Self {
        Self::Original { node }
    }
}

/// A wrapper on Edge which lets us mark a node as contracted during a particular iteration.
///
/// Rather than allocating a new graph on each contraction iteration, we can simply annotate at
/// which iteration each shortcut was formed or edge orphaned.
#[derive(Clone)]
pub enum CHEdge<Edge> {
    /// An original edge from the input graph, unmodified.
    Original {
        /// The original edge data
        edge: Edge,
    },
    /// A shortcut edge formed by merging multiple original edges when their shared node is
    /// contracted.
    Shortcut {
        /// The original edges that were merged to form the shortcut.
        edges: Vec<Edge>,
        /// The nodes that were merged to form the shortcut.
        nodes: Vec<NodeIndex>,
        /// The iteration number when the shortcut was formed.
        iteration: usize,
    },
    /// An edge that has been orphaned because one of its endpoints was contracted.
    Orphaned {
        /// The original edge data
        edge: Edge,
        /// The iteration number when the edge was orphaned.
        iteration: usize,
    },
}

impl<Edge> CHEdge<Edge> {
    /// Creates a new `CHEdge` in the Original state.
    const fn new_original(edge: Edge) -> Self {
        Self::Original { edge }
    }
}

// Custom Debug implementation for CHEdge to show probability of shortcut
impl<Edge: std::fmt::Debug + Distance> std::fmt::Debug for CHEdge<Edge> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Original { edge } => f.debug_struct("Original").field("edge", edge).finish(),
            Self::Shortcut { edges, nodes, iteration } => {
                let probability = self.probability().into_inner(); // Calculate probability from surprisal

                f.debug_struct("Shortcut")
                    .field("edges", edges)
                    .field("nodes", nodes)
                    .field("iteration", iteration)
                    .field("probability", &probability)
                    .finish()
            },
            Self::Orphaned { edge, iteration } => f
                .debug_struct("Orphaned")
                .field("edge", edge)
                .field("iteration", iteration)
                .finish(),
        }
    }
}

/// Trait for objects that have an associated probability.
/// This is used to collect probability/surprisal logic to prevent code duplication.
///
/// The probability of multiple edges can be combined by multiplying their probabilities. The
/// surprisal of multiple edges can be combined by summing their surprisals. To satisfy metric space
/// assumptions, we use surprisal in place of distances in shortest path algorithms. Using surprisal
/// for "shortest path" search is equivalent to finding the lowest surprisal path, which is the
/// highest probability path.
pub trait Distance {
    /// Returns the probability value associated with this item.
    ///
    /// This represents a normalized probability value between 0 and 1.
    fn probability(&self) -> OrderedFloat<f64>;

    /// Calculates the surprisal (negative log probability) of this item.
    fn surprisal(&self) -> OrderedFloat<f64> {
        -self.probability().ln()
    }
}


impl<E: Distance> Distance for CHEdge<E> {
    fn probability(&self) -> OrderedFloat<f64> {
        match self {
            Self::Original { edge } => edge.probability(),
            Self::Shortcut { edges, .. } => edges.iter().map(Distance::probability).product(),
            Self::Orphaned { .. } => OrderedFloat(0.0),
        }
    }

    fn surprisal(&self) -> OrderedFloat<f64> {
        match self {
            Self::Original { edge } => edge.surprisal(),
            Self::Shortcut { edges, .. } => edges.iter().map(Distance::surprisal).sum(),
            Self::Orphaned { .. } => OrderedFloat(f64::INFINITY),
        }
    }
}

impl Distance for () {
    fn probability(&self) -> OrderedFloat<f64> {
        OrderedFloat(1.0)
    }
}

/// Trait for determining the order in which nodes should be contracted.
///
/// Implementations of this trait provide a strategy for selecting the next node
/// to contract in the graph during the contraction hierarchy construction.
pub trait ContractionHeuristic<N, E> {
    /// Returns the next node to contract based on the current graph state.
    ///
    /// Should only be called after the previously specified node (if any) has been contracted.
    fn next_contraction(&mut self, graph: &Graph<CHNode<N>, CHEdge<E>>) -> Option<NodeIndex>;
}

impl<I, N, E> ContractionHeuristic<N, E> for I
where
    I: Iterator<Item = NodeIndex>,
{
    fn next_contraction(&mut self, _graph: &Graph<CHNode<N>, CHEdge<E>>) -> Option<NodeIndex> {
        self.next()
    }
}

/// Result of a shortest path search between two nodes.
///
/// Contains the path information (nodes and edges) as well as the total cost (surprisal).
#[derive(Clone)]
struct SearchResult {
    /// The total surprisal (negative log probability) of the path.
    surprisal: ordered_float::OrderedFloat<f64>,
    /// The sequence of nodes in the path.
    path: Vec<NodeIndex>,
    /// The sequence of edges in the path.
    path_edges: Vec<EdgeIndex>,
}

// Y-Statement ADR on the design of CH, CHNode, and CHEdge:
// In the context of designing a type to represent the state of the Contraction Hierarchies
// algorithm, where the output graph--the eponymous "contraction hierarchy"-- is the union of all
// intermediate states--"core graphs", facing the need to minimize memory footprint while retaining
// conceptual simplicity, we decided for a representation which wraps the original Node and Edge
// types to optionally mark each as contracted or as a shortcut during a particular iteration on a
// single continuously increasing graph and neglected recording each core graph separately--where
// contracted nodes are deleted and shortcuts indistiguishable from original edges-- to achieve the
// reduced memory footprint of a single representation from which both all prior core graphs and the
// final contraction hierarchy can be cheaply computed, accepting the increased cost during each
// witness search of having to check (and subsequently skip) edges to contracted nodes, because this
// cost is bounded by the initial degrees of each node, which are constant, and other mitigating
// steps (such as storing edges to contracted nodes separately or later in the adjacency list) are
// concievable

// Remove IntoIterator complexity and just use a generic type H that implements ContractionHeuristic
#[derive(Clone)]
/// A data structure that wraps a graph of node type `N` and edge type `E` with a contraction
/// heuristic `H`.
///
/// Contraction hierarchies enable efficient path finding by preprocessing a graph,
/// adding shortcuts when nodes are contracted, and preserving shortest path distances.
///
/// We use a single graph to represent the entire contraction hierarchy, annotating each node and
/// edge with additional information rather than creating a new graph on each iteration.
///
/// Type parameters:
/// - `N`: Node data type
/// - `E`: Edge data type implementing the `Distance` trait
/// - `H`: Contraction heuristic that determines the order of node contractions
pub struct CH<N, E, H>
where
    E: Distance,
    H: ContractionHeuristic<N, E>,
{
    /// A graph of node type `N` and edge type `E` with annotations indicating which nodes and edges
    /// are contracted or orphaned.
    graph: Graph<CHNode<N>, CHEdge<E>>,
    /// A function which returns the next node to contract given the current graph state (which may
    /// be ignored by the heuristic if it is not stateful).
    heuristic: H,
    /// The number of contractions performed so far.
    num_contractions: usize,
}

use std::fmt::Debug;

impl<N: Clone + Hash + Eq + Debug, E: Clone + Hash + Debug, H> CH<N, E, H>
where
    N: Clone + Hash + Eq + Debug,
    E: Clone + Hash + Debug + Distance,
    H: ContractionHeuristic<N, E>,
{
    /// Annotate a graph over node type `N` and edge type `E` with `CHNode` and `CHEdge` wrappers.
    fn annotate_graph(graph: Graph<N, E>) -> Graph<CHNode<N>, CHEdge<E>> {
        graph.map(|_, n| CHNode::new_original(n.clone()), |_, e| CHEdge::new_original(e.clone()))
    }

    /// Create a new contraction hierarchy from an input graph and a contraction heuristic.
    pub fn new(graph: Graph<N, E>, heuristic: H) -> Self {
        let graph = Self::annotate_graph(graph);

        Self { graph, heuristic, num_contractions: 0 }
    }

    /// Compute the shortest path between two nodes in the contraction hierarchy.
    ///
    /// This method uses Dijkstra's algorithm to find the shortest path between two nodes in the
    /// graph. It skips edges that connect to contracted nodes or are orphaned.
    fn g_distance(&self, x_index: NodeIndex, y_index: NodeIndex) -> Option<SearchResult> {
        let result = dijkstra(&self.graph, x_index, Some(y_index), |e| {
            use petgraph::visit::EdgeRef;
            // Skip edges that connect to contracted nodes or are orphaned
            match (&self.graph[e.source()], &self.graph[e.target()], &self.graph[e.id()]) {
                (CHNode::Contracted { .. }, ..) | (_, CHNode::Contracted { .. }, _) => OrderedFloat::nan(),
                (_, _, CHEdge::Orphaned { .. }) => {
                    OrderedFloat::nan() // Skip orphaned edges too
                },
                _ => e.weight().surprisal(),
            }
        });

        // Reconstruct path from predecessors map
        let mut path = Vec::new();
        let mut path_edges = Vec::new();
        let mut current = y_index;
        path.push(current);

        while let Some(&prev) = result.predecessors.get(&current) {
            // Find the edge between prev and current
            if let Some(edge_id) = self.graph.find_edge(prev, current) {
                path_edges.push(edge_id);
            }
            
            path.push(prev);
            current = prev;
            if current == x_index {
                break;
            }
        }
        path.reverse();
        path_edges.reverse();

        // Only return Some if the distance is less than infinity (meaning a valid path was found).
        result.distances.get(&y_index).and_then(|&distance| {
            // TODO validate this use of ordered float is correct
            if distance < OrderedFloat::infinity() {
                Some(SearchResult { 
                    surprisal: distance,
                    path,
                    path_edges,
                })
            } else {
                None
            }
        })
    }

    /// Limited-distance version of `g_distance`.
    ///
    /// Intended to constrain the search to paths with surprisal less than the limit.
    /// Currently just delegates to `g_distance` without using the limit.
    fn g_distance_limited(
        &self,
        x: NodeIndex,
        y: NodeIndex,
        #[allow(unused_variables)]
        limit: ordered_float::OrderedFloat<f64>,
    ) -> Option<SearchResult> {
        // TODO actually use limit
        self.g_distance(x, y)
    }

    /// Contracts a node in the graph, adding necessary shortcuts to preserve distances.
    ///
    /// This is the core operation of the contraction hierarchies algorithm. When a node is 
    /// contracted, it is removed from the graph and shortcuts are added between its neighbors
    /// to preserve the shortest path distances that went through the contracted node.
    fn contract(&mut self, node_index: NodeIndex) {
        // "To compute G′, one iterates over all pairs of neighbors x, y of v increasing by distG(x, y)."
        use petgraph::visit::EdgeRef;
        use petgraph::Direction;

        // First, collect all incident edges before we change any of them
        let incoming_edges: Vec<_> = self
            .graph
            .edges_directed(node_index, Direction::Incoming)
            .map(|edge| (edge.id(), edge.source()))
            .collect();

        let outgoing_edges: Vec<_> = self
            .graph
            .edges_directed(node_index, Direction::Outgoing)
            .map(|edge| (edge.id(), edge.target()))
            .collect();

        let out_neighbors = self
            .graph
            .neighbors_directed(node_index, Direction::Outgoing)
            .filter(|&n| !matches!(self.graph[n], CHNode::Contracted { .. }));
        let in_neighbors = self
            .graph
            .neighbors_directed(node_index, Direction::Incoming)
            .filter(|&n| !matches!(self.graph[n], CHNode::Contracted { .. }));


        // First collect all pairs and their original distances before contraction
        let in_out_pairs: Vec<_> = Itertools::cartesian_product(in_neighbors, out_neighbors)
            .map(|(x, y)| {
                (
                    x,
                    y,
                    self.g_distance(x, y)
                        .with_context(|| {
                            format!("Failed to compute distance between {:?} and {:?} on graph {:#?}", x, y, self.graph)
                        })
                        .unwrap()
                        .surprisal,
                )
            })
            .sorted_by_key(|(_, _, d)| *d)
            .collect();

        // Now mark the node as contracted
        match &self.graph[node_index] {
            CHNode::Original { node } => {
                self.graph[node_index] = CHNode::Contracted {
                    node: node.clone(),
                    iteration: self.num_contractions,
                };
            },
            CHNode::Contracted { .. } => {
                panic!("Attempted to contract node {node_index:?} which is already contracted");
            },
        }

        // Mark all incident edges as orphaned
        for (edge_id, _) in &incoming_edges {
            if let CHEdge::Original { edge } = self.graph[*edge_id].clone() {
                self.graph[*edge_id] = CHEdge::Orphaned { edge, iteration: self.num_contractions };
            }
        }

        for (edge_id, _) in &outgoing_edges {
            if let CHEdge::Original { edge } = self.graph[*edge_id].clone() {
                self.graph[*edge_id] = CHEdge::Orphaned { edge, iteration: self.num_contractions };
            }
        }

        // witness search -- i.e. does removing v destroy the previously existing shortest path between x
        // and y? TODO: Shortcut should probability sum over all path lengths to preserve stochastic
        // transition probabilities       There may be a better algorithm for "find probability of
        // all probability-weighted paths from A->C via B"

        for (x, y, d) in in_out_pairs {
            // TODO: We probably need to search the whole graph to avoid looking at any paths
            let search_result = self.g_distance_limited(x, y, d * 2.0);

            let should_add_shortcut = match search_result.clone() {
                Some(result) if result.surprisal <= d => false,
                Some(_) | None => true,
            };

            if should_add_shortcut {
                debug!("Adding shortcut from {:?} to {:?}", x, y);
                let mut path_edges = Vec::new();
                let path_nodes;
                
                if let Some(result) = &search_result {
                    path_nodes = result.path.clone();
                    
                    for &edge_idx in &result.path_edges {
                        match &self.graph[edge_idx] {
                            CHEdge::Original { edge } | CHEdge::Orphaned { edge, .. } => {
                                path_edges.push(edge.clone());
                            },
                            CHEdge::Shortcut { edges, .. } => {
                                // If the edge is already a shortcut, add all its component edges
                                path_edges.extend(edges.iter().cloned());
                            },
                        }
                    }
                } else {
                    info!("Failed to find path from {:?} to {:?}, falling back to a basic path through the contracted node", x, y);
                    path_nodes = vec![x, node_index, y];
                    
                    // Find the edges forming the path through the contracted node (x -> node_index -> y)
                    // Get edge from x to node_index
                    if let Some(in_edge_idx) = self.graph.find_edge(x, node_index) {
                        // Need to clone the edge weight to get its original value before potential orphaning
                        let edge_weight = self.graph[in_edge_idx].clone();
                        match edge_weight {
                            CHEdge::Original { edge } | CHEdge::Orphaned { edge, .. } => {
                                path_edges.push(edge);
                            },
                            CHEdge::Shortcut { edges, .. } => {
                                // If the incoming edge is already a shortcut, use its aggregated surprisal
                                path_edges.extend(edges); // Keep original edges if needed
                            },
                        }
                    }

                    if let Some(out_edge_idx) = self.graph.find_edge(node_index, y) {
                        let edge_weight = self.graph[out_edge_idx].clone();
                        match edge_weight {
                            CHEdge::Original { edge } | CHEdge::Orphaned { edge, .. } => {
                                path_edges.push(edge);
                            },
                            CHEdge::Shortcut { edges, .. } => {
                                path_edges.extend(edges);
                            },
                        }
                    }
                }

                // Do we already have an edge x → y (that is not orphaned)?
                if let Some(eidx) = self
                    .graph
                    .find_edge(x, y)
                    .filter(|&e| !matches!(self.graph[e], CHEdge::Orphaned { .. }))
                {
                    if let CHEdge::Shortcut { edges, nodes, .. } = &mut self.graph[eidx] {
                        // Update existing Shortcut
                        *edges = path_edges;
                        *nodes = path_nodes;
                    } else {
                        // Existing edge is Original - convert it to a Shortcut
                        // Clone the original edge data before overwriting
                        if let CHEdge::Original { .. } = self.graph[eidx].clone() {
                            *self.graph.edge_weight_mut(eidx).unwrap() = CHEdge::Shortcut {
                                edges: path_edges,
                                nodes: path_nodes,
                                iteration: self.num_contractions,
                            };
                        } else {
                            error!("Found non-Original, non-Orphaned edge that wasn't a Shortcut during merge at iteration {}", self.num_contractions);
                            // Fallback: create a new shortcut anyway? Or panic?
                            // For now, let's overwrite with a new shortcut based on w_add
                            *self.graph.edge_weight_mut(eidx).unwrap() = CHEdge::Shortcut {
                                edges: path_edges,
                                nodes: path_nodes,
                                iteration: self.num_contractions,
                            };
                        }
                    }
                } else {
                    // --- no edge yet: create one ---
                    self.graph.add_edge(
                        x,
                        y,
                        CHEdge::Shortcut {
                            edges: path_edges,
                            nodes: path_nodes,
                            iteration: self.num_contractions,
                        },
                    );
                }
            }
        }
        self.num_contractions += 1;
    }

    /// Contracts the graph up to the specified iteration.
    ///
    /// This method repeatedly calls `contract` with the next node determined by the heuristic
    /// until the specified iteration count is reached.
    fn contract_to(&mut self, iteration: usize) -> Result<&mut Self> {
        while self.num_contractions < iteration {
            let next_contraction = self
                .heuristic
                .next_contraction(&self.graph)
                .context("No more contractions to perform")?;
            self.contract(next_contraction);
        }
        Ok(self)
    }

    /// Contracts the graph up to the specified iteration with progress reporting.
    ///
    /// Similar to `contract_to` but calls the provided callback function after each
    /// contraction with the current iteration number.
    fn contract_to_with_progress<F>(&mut self, iteration: usize, mut progress_callback: F) -> Result<&mut Self>
    where
        F: FnMut(usize),
    {
        while self.num_contractions < iteration {
            let next_contraction = self
                .heuristic
                .next_contraction(&self.graph)
                .context("No more contractions to perform")?;
            self.contract(next_contraction);
            progress_callback(self.num_contractions);
        }
        Ok(self)
    }

    /// Returns the core graph at contraction iteration `i`.
    ///
    /// This contracts the graph up to the specified iteration and returns
    /// a filtered version representing the state at that point.
    pub fn core_graph(&mut self, i: usize) -> Result<Graph<CHNode<N>, CHEdge<E>>>
    where
        N: Clone,
        E: Clone,
    {
        self.contract_to(i)?;

        Ok(self.graph.filter_map(
            |_, n| match n {
                CHNode::Original { node } => Some(CHNode::Original { node: node.clone() }),
                CHNode::Contracted { node, iteration } => {
                    if *iteration > i {
                        Some(CHNode::Original { node: node.clone() })
                    } else {
                        Some(n.clone())
                    }
                },
            },
            |_, e| match e {
                CHEdge::Original { .. } | CHEdge::Orphaned { .. } => Some(e.clone()),
                CHEdge::Shortcut { iteration, .. } => {
                    if *iteration <= i {
                        Some(e.clone())
                    } else {
                        None
                    }
                },
            },
        ))
    }

    /// Returns the core graph at contraction iteration `i` with progress reporting.
    ///
    /// This contracts the graph up to the specified iteration and returns
    /// a filtered version representing the state at that point.
    ///
    /// The `progress_callback` is called after each contraction with the current iteration number.
    pub fn core_graph_with_progress<F>(&mut self, i: usize, progress_callback: F) -> Result<Graph<CHNode<N>, CHEdge<E>>>
    where
        N: Clone,
        E: Clone,
        F: FnMut(usize),
    {
        self.contract_to_with_progress(i, progress_callback)?;

        Ok(self.graph.filter_map(
            |_, n| match n {
                CHNode::Original { node } => Some(CHNode::Original { node: node.clone() }),
                CHNode::Contracted { node, iteration } => {
                    if *iteration > i {
                        Some(CHNode::Original { node: node.clone() })
                    } else {
                        Some(n.clone())
                    }
                },
            },
            |_, e| match e {
                CHEdge::Original { .. } | CHEdge::Orphaned { .. } => Some(e.clone()),
                CHEdge::Shortcut { iteration, .. } => {
                    if *iteration <= i {
                        // TODO check off-by-one
                        Some(e.clone())
                    } else {
                        None
                    }
                },
            },
        ))
    }

    /// Returns the complete contraction hierarchy after contracting all remaining nodes.
    ///
    /// A contraction hierarchy is the union of all core graphs - the original graph
    /// with all shortcuts added during the contraction process. This method contracts
    /// all nodes in the graph based on the heuristic and returns a filtered version
    /// with original node representation.
    pub fn contraction_hierarchy(&mut self) -> Result<Graph<CHNode<N>, CHEdge<E>>> {
        while let Some(next_contraction) = self.heuristic.next_contraction(&self.graph) {
            self.contract(next_contraction);
        }

        Ok(self.graph.filter_map(
            |_, n| match n {
                CHNode::Original { node } | CHNode::Contracted { node, .. } => {
                    Some(CHNode::Original { node: node.clone() })
                },
            },
            |_, e| Some(e.clone()), // Include all edges, including orphaned edges
        ))
    }
}

/// Trait for computing probability measures on a graph.
///
/// This trait provides methods for calculating probabilities of edges and paths
/// in a graph with edge weights that implement the `Distance` trait.
pub trait GraphDistance<E: Distance> {
    /// Returns the probability of traversing a single edge in the graph.
    fn edge_probability(&self, edge_idx: &EdgeIndex) -> OrderedFloat<f64>;
    
    /// Returns the probability of traversing a sequence of edges in the graph.
    ///
    /// The probability of a path is the product of the probabilities of its edges.
    #[allow(dead_code)]
    fn path_probability(&self, edge_indices: &[EdgeIndex]) -> OrderedFloat<f64>;
}

// implement GraphDistance for any Graph with edge weights implementing Distance
impl<N, E: Distance> GraphDistance<E> for Graph<N, E> {
    fn edge_probability(&self, edge_idx: &EdgeIndex) -> OrderedFloat<f64> {
        self.edge_weight(*edge_idx).expect("Edge index should be valid").probability()
    }

    fn path_probability(&self, edge_indices: &[EdgeIndex]) -> OrderedFloat<f64> {
        edge_indices.iter().map(|&idx| self.edge_probability(&idx)).product()
    }
}
