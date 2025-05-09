//! Implementation of the synthetic trace generation pipeline.
//!
//! This module orchestrates the four main phases of trace generation:
//! 1. Graph construction - Convert observed traces into a graph of cluster states
//! 2. Graph expansion - Apply synthetic transitions to expand the state space
//! 3. Graph contraction - Use contraction hierarchies to shrink the graph and create shortcuts to
//!    preserve the most probable paths
//! 4. Trace generation - Sample random walks through the graph to create synthetic traces
//!
//! Each phase is implemented as a separate function, coordinated by the `run` function.

use std::collections::{
    HashMap,
    HashSet,
};
use std::path::{
    Path,
    PathBuf,
};

use anyhow::Result;
use indicatif::{
    ParallelProgressIterator,
    ProgressBar,
    ProgressFinish,
    ProgressStyle,
};
use ordered_float::OrderedFloat;
use petgraph::dot::Dot;
use petgraph::prelude::*;
use rand::distributions::{
    Distribution,
    WeightedIndex,
};
use rand::thread_rng;
use rayon::prelude::*;
use sk_store::TraceEvent;
use tracing::{
    info,
    instrument,
    warn,
};

use crate::contraction_hierarchies::{
    CHEdge,
    CHNode,
    NestedDissection,
};
use crate::model::{
    Action,
    Edge,
    Node,
    ObjectKey,
    ObjectType,
};
use crate::utils::{
    create_timestamped_output_dir,
    diff_objects,
    tracestore_from_events,
    write_dot_file,
};


/// Public entry-point that orchestrates the full pipeline.
/// End-to-end run of the synthetic trace generation pipeline.
#[instrument(skip(next_action_fn, input_traces), fields(input_traces_count = input_traces.len()))]
pub fn run<F>(
    next_action_fn: F,
    input_traces: Vec<Vec<TraceEvent>>,
    num_samples: usize,
    trace_length: u64,
    enumeration_steps: u64,
    contraction_strength: f64,
) -> Result<()>
where
    F: Fn(&Node) -> Vec<Action> + Clone + Sync,
{
    let output_dir = create_timestamped_output_dir()?;

    // Use first event of first trace as the common starting state (may be None if all have been
    // contracted).
    let first_trace_event = input_traces.first().and_then(|trace| trace.first()).cloned();
    if first_trace_event.is_none() {
        warn!(
            "No initial trace event found in input traces. Generated traces will not have a consistent starting state."
        );
    }

    let (mut state_graph, mut node_to_index) = construct_graph(input_traces)?;
    write_graph_visualization(&state_graph, &output_dir, "initial_graph.dot", "Initial graph from traces")?;

    expand_graph(&mut state_graph, &mut node_to_index, next_action_fn, enumeration_steps)?;
    write_graph_visualization(&state_graph, &output_dir, "expanded_graph.dot", "Expanded graph after enumeration")?;

    let processed_graph = contract_graph(state_graph, &output_dir, contraction_strength)?;
    write_graph_visualization(
        &processed_graph,
        &output_dir,
        "contracted_graph.dot",
        "Final contracted graph for sampling",
    )?;
    write_graph_visualization_with_probabilities(
        &processed_graph,
        &output_dir,
        "processed_sample_graph.dot",
        "Processed graph with probability details",
    )?;

    generate_traces(&processed_graph, &output_dir, num_samples, trace_length, first_trace_event.as_ref())?;
    Ok(())
}

/// Phase 1 – construct a state graph from the observed input traces.
#[instrument(skip(input_traces), fields(input_traces_count = input_traces.len()))]
pub(crate) fn construct_graph(
    input_traces: Vec<Vec<TraceEvent>>,
) -> Result<(DiGraph<Node, Edge>, HashMap<Node, NodeIndex>)> {
    let mut graph = DiGraph::new();
    let mut node_to_index: HashMap<Node, NodeIndex> = HashMap::new();

    for trace in input_traces {
        let mut iter = trace.into_iter();
        let Some(first_event) = iter.next() else { continue }; // empty trace

        for deleted_obj in &first_event.deleted_objs {
            warn!(?deleted_obj, "Ignoring deleted object in first event of trace");
        }

        let objects = first_event
            .applied_objs
            .iter()
            .map(|obj| (ObjectKey::from(obj), obj.clone().into()))
            .collect();

        let mut current_node = Node {
            object_type: ObjectType::Observed,
            objects,
            ts: first_event.ts,
        };

        for event in iter {
            let next_node = current_node.apply_patch(&event)?;

            let edge = Edge {
                object_type: ObjectType::Observed,
                action: Action {
                    trace_event_newtype: event.into(),
                    probability: OrderedFloat(1.0),
                    message: Some("generated from trace".to_string()),
                },
            };

            let current_idx = *node_to_index
                .entry(current_node.clone())
                .or_insert_with(|| graph.add_node(current_node.clone()));
            let next_idx = *node_to_index
                .entry(next_node.clone())
                .or_insert_with(|| graph.add_node(next_node.clone()));

            graph.update_edge(current_idx, next_idx, edge);
            current_node = next_node;
        }
    }
    Ok((graph, node_to_index))
}

/// Normalize the probabilities of edges in the graph such that the sum of the probabilities of all
/// outgoing edges from a node is 1.
#[instrument(skip(graph), fields(nodes = graph.node_count(), edges = graph.edge_count()))]
fn normalize_edge_probabilities(graph: &mut DiGraph<Node, Edge>) {
    use petgraph::visit::EdgeRef;
    use petgraph::Direction;

    for node_idx in graph.node_indices() {
        let outgoing: Vec<_> = graph
            .edges_directed(node_idx, Direction::Outgoing)
            .map(|e| (e.id(), e.weight().action.probability.into_inner()))
            .collect();
        if outgoing.is_empty() {
            continue;
        }
        let total: f64 = outgoing.iter().map(|(_, p)| *p).sum();
        if total == 0.0 {
            continue;
        }
        for (edge_id, prob) in outgoing {
            if let Some(weight) = graph.edge_weight_mut(edge_id) {
                weight.action.probability = OrderedFloat(prob / total);
            }
        }
    }
}

/// Phase 2 – expand the graph with synthetic actions generated by `next_action_fn`.
#[instrument(skip(graph, node_to_index, next_action_fn), fields(initial_nodes = graph.node_count()))]
pub(crate) fn expand_graph<F>(
    graph: &mut DiGraph<Node, Edge>,
    node_to_index: &mut HashMap<Node, NodeIndex>,
    next_action_fn: F,
    enumeration_steps: u64,
) -> Result<()>
where
    F: Fn(&Node) -> Vec<Action> + Clone + Sync,
{
    let mut current_layer: Vec<NodeIndex> = graph.node_indices().collect();
    let mut next_layer: Vec<NodeIndex> = Vec::new();
    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut depth = 0;

    while !current_layer.is_empty() && depth < enumeration_steps {
        info!(depth, layer_nodes = current_layer.len(), "Expanding graph");

        // Filter out nodes that have already been visited in previous layers so we don't
        // over-estimate the amount of work left and show an inaccurate progress bar.
        let current_copy = std::mem::take(&mut current_layer);
        let to_process: Vec<_> = current_copy.into_iter().filter(|idx| visited.insert(*idx)).collect();

        let total_nodes = to_process.len();

        // Style we want for the iterator-driven progress bar.
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} nodes ({percent}%) {msg}")
            .unwrap();


        let results: Vec<_> = to_process
            .par_iter()
            .progress_count(total_nodes as u64)
            .with_style(style)
            .with_message(format!("Layer {}/{}", depth + 1, enumeration_steps))
            .with_finish(ProgressFinish::AndLeave)
            .map(|&current_idx| {
                let current_node_data = graph.node_weight(current_idx).cloned().expect("node not found");
                let actions = next_action_fn(&current_node_data);

                let mut res = Vec::new();
                for action in actions {
                    let next_node_data = match current_node_data.apply_patch(&action.trace_event_newtype) {
                        Ok(mut n) => {
                            n.object_type = ObjectType::Synthetic;
                            n
                        },
                        Err(e) => {
                            warn!(?e, "Skipping invalid action");
                            continue;
                        },
                    };
                    res.push((next_node_data, action, current_idx));
                }
                (current_idx, res)
            })
            .collect();

        for (current_idx, node_results) in results {
            for (next_node_data, action, _) in node_results {
                let next_idx = *node_to_index
                    .entry(next_node_data.clone())
                    .or_insert_with(|| graph.add_node(next_node_data));
                graph.update_edge(current_idx, next_idx, Edge { object_type: ObjectType::Synthetic, action });
                if !visited.contains(&next_idx) && (depth + 1) < enumeration_steps {
                    next_layer.push(next_idx);
                }
            }
        }

        depth += 1;
        current_layer = std::mem::take(&mut next_layer);
    }

    normalize_edge_probabilities(graph);
    Ok(())
}

/// Phase 3 – contract the expanded graph using Contraction Hierarchies.
#[instrument(skip(state_graph))]
pub(crate) fn contract_graph(
    state_graph: DiGraph<Node, Edge>,
    output_dir: &Path,
    contraction_strength: f64,
) -> Result<DiGraph<Node, Edge>> {
    let total_nodes = state_graph.node_count();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )] // product of two positives is positive so cast to usize is safe
    let target = ((total_nodes as f64) * contraction_strength).round() as usize;


    let contraction_order = NestedDissection::contraction_order(&state_graph);
    let iterations = target.min(contraction_order.len());

    info!(iterations, total_nodes, contraction_strength, "Starting contraction");
    let mut ch = crate::contraction_hierarchies::CH::new(&state_graph, contraction_order.into_iter());

    let pb = ProgressBar::new(iterations as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} contractions ({percent}%) {msg}",
                )
                .unwrap()
        )
        .with_message(format!("Contraction phase ({:.0}% of nodes)", contraction_strength * 100.0))
        .with_finish(ProgressFinish::AndLeave);
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut contracted = 0;
    let core_graph = ch.core_graph_with_progress(iterations, |i| {
        contracted = i;
        pb.set_position(i as u64);
    })?;
    pb.finish_using_style();

    // Convert CH graph back to plain `DiGraph<Node, Edge>` while combining shortcut edges.
    let processed = core_graph.filter_map(
        |_, node| match node {
            CHNode::Original { node } | CHNode::Contracted { node, .. } => Some(node.clone()),
        },
        |_edge_idx, edge| match edge {
            CHEdge::Original { edge } => Some(edge.clone()),
            CHEdge::Shortcut { edges, nodes, .. } => {
                // Combine underlying edges into a single composite edge.
                let messages = edges.iter().filter_map(|e| e.action.message.clone()).collect::<Vec<_>>();
                let message_opt = (!messages.is_empty()).then(|| messages.join(" -> "));

                let start_node_idx = nodes.first()?;
                let start_node = match &core_graph[*start_node_idx] {
                    CHNode::Original { node } | CHNode::Contracted { node, .. } => node,
                };
                let mut state = start_node.clone();
                let mut final_ts = state.ts;
                for original in edges {
                    state = state.apply_patch(&original.action.trace_event_newtype).ok()?;
                    final_ts = state.ts;
                }
                let (applied_objs, deleted_objs) = diff_objects(&start_node.objects, &state.objects);
                let combined_patch = TraceEvent { ts: final_ts, applied_objs, deleted_objs };
                let prob: f64 = edges.iter().map(|e| e.action.probability.into_inner()).product();

                Some(Edge {
                    object_type: ObjectType::Synthetic,
                    action: Action {
                        trace_event_newtype: combined_patch.into(),
                        probability: prob.into(),
                        message: message_opt,
                    },
                })
            },
            CHEdge::Orphaned { .. } => None,
        },
    );

    Ok(processed)
}

/// Phase 4 – sample random walks over the processed graph and emit synthetic trace files.
pub(crate) fn generate_traces(
    graph: &DiGraph<Node, Edge>,
    output_dir: &Path,
    num_samples: usize,
    trace_length: u64,
    initial_event: Option<&TraceEvent>,
) -> Result<()> {
    let start_node = graph
        .node_indices()
        .find(|&idx| graph.neighbors(idx).next().is_some())
        .ok_or_else(|| anyhow::anyhow!("No node with outgoing edges found in graph"))?;

    let adjusted_len = if initial_event.is_some() { trace_length } else { trace_length + 1 };
    let walks = walks_with_sampling(graph, start_node, adjusted_len, num_samples);

    let pb = ProgressBar::new(walks.len() as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} traces ({percent}%) {msg}",
                )
                .unwrap(),
        )
        .with_message(format!("Generating traces (length {trace_length})"))
        .with_finish(ProgressFinish::AndLeave);
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let counter = std::sync::atomic::AtomicUsize::new(0);

    walks.par_iter().enumerate().try_for_each(|(i, walk)| -> Result<()> {
        let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count % 10 == 0 || count == walks.len() - 1 {
            pb.set_position(count as u64);
        }

        if walk.len() <= 1 {
            return Ok(());
        }
        let path = output_dir.join(format!("trace_{i}.json"));
        generate_trace_file(graph, walk, &path, initial_event.cloned())?;
        Ok(())
    })?;
    pb.finish_using_style();
    info!("Trace generation complete");
    Ok(())
}

/// Helper used by `generate_traces`: sample multiple random walks with probability-weighted edges.
pub(crate) fn walks_with_sampling(
    graph: &DiGraph<Node, Edge>,
    start_node: NodeIndex,
    walk_length: u64,
    num_samples: usize,
) -> Vec<Vec<NodeIndex>> {
    use rand::distributions::WeightedIndex;
    use rand::thread_rng;

    (0..num_samples)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut walk = vec![start_node];
            let mut current = start_node;
            for _step in 1..walk_length {
                let neighbors: Vec<_> = graph.neighbors(current).collect();
                if neighbors.is_empty() {
                    break;
                }
                let weights: Vec<f64> = neighbors
                    .iter()
                    .filter_map(|&n| graph.find_edge(current, n))
                    .filter_map(|eidx| graph.edge_weight(eidx))
                    .map(|e| e.action.probability.into_inner().max(0.0))
                    .collect();
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    break;
                }
                let dist = WeightedIndex::new(&weights).expect("WeightedIndex failure");
                let next_idx = neighbors[dist.sample(&mut rng)];
                walk.push(next_idx);
                current = next_idx;
            }
            walk
        })
        .collect()
}

#[instrument(skip(graph, walk, initial_event))]
fn generate_trace_file(
    graph: &DiGraph<Node, Edge>,
    walk: &[NodeIndex],
    path: &Path,
    initial_event: Option<TraceEvent>,
) -> Result<()> {
    let mut events = Vec::new();
    if let Some(ev) = initial_event {
        events.push(ev);
    }
    let mut next_ts: i64 = events.last().map_or(0, |e| e.ts);

    for window in walk.windows(2) {
        let u = window[0];
        let v = window[1];
        let edge_index = graph
            .find_edge(u, v)
            .ok_or_else(|| anyhow::anyhow!("Edge missing between consecutive nodes"))?;
        let edge = graph
            .edge_weight(edge_index)
            .ok_or_else(|| anyhow::anyhow!("Edge weight missing"))?;
        let mut ev = edge.action.trace_event_newtype.trace_event.clone();
        next_ts += 1;
        ev.ts = next_ts;
        events.push(ev);
    }
    if events.is_empty() {
        anyhow::bail!("Generated walk resulted in zero trace events");
    }

    let store = tracestore_from_events(&events);
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(&file, &store)?;
    Ok(())
}

//  Utility – DOT visualisation helpers

#[instrument(skip(graph))]
fn write_graph_visualization(
    graph: &DiGraph<Node, Edge>,
    output_dir: &Path,
    filename: &str,
    description: &str,
) -> Result<PathBuf> {
    let graphable = graph.map(
        |i, n| format!("{} -- {:?}", i.index(), n.object_type),
        |i, e| format!("{} -- {:?} {}", i.index(), e.object_type, e.action.message.as_deref().unwrap_or("")),
    );
    let dot = Dot::new(&graphable);
    let dot_str = format!("{dot}");
    let path = write_dot_file(output_dir, filename, &dot_str)?;
    info!("{} written to: {}", description, path.display());
    Ok(path)
}

#[instrument(skip(graph))]
fn write_graph_visualization_with_probabilities(
    graph: &DiGraph<Node, Edge>,
    output_dir: &Path,
    filename: &str,
    description: &str,
) -> Result<PathBuf> {
    let graphable = graph.map(
        |i, n| format!("{} -- {:?}", i.index(), n.object_type),
        |i, e| {
            format!(
                "{} -- {:?} P={:.2e} {}",
                i.index(),
                e.object_type,
                e.action.probability.into_inner(),
                e.action.message.as_deref().unwrap_or("")
            )
        },
    );
    let dot = Dot::new(&graphable);
    let dot_str = format!("{dot}");
    let path = write_dot_file(output_dir, filename, &dot_str)?;
    info!("{} written to: {}", description, path.display());
    Ok(path)
}
