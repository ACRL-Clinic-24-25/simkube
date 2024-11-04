#![deny(rustdoc::broken_intra_doc_links)]
//! `sk-gen` is a CLI tool for generating synthetic trace data for SimKube.
//!
//! # Overview:
//! ## Core types
//! [`Node`] represents a cluster state, containing a map from unique names to active
//! [`Deployment`] states. `Node` implements [`Eq`] and [`Hash`], which we use to ensure that
//! equivalent `Node`s are not duplicated in the graph.
//!
//! [`Deployment`] is a simplified representation of a Kubernetes deployment spec, containing only
//! the fields we are considering.
//!
//! [`DeploymentAction`] (e.g. `CreateDeployment`, `DeleteDeployment`, `IncrementReplicas`,
//! `DecrementReplicas`) can be performed on individual deployment instances.
//!
//! [`ClusterAction`] contains a name of a candidate deployment alongside a [`DeploymentAction`]
//! such that it can be applied to a `Node` without ambiguity as to which deployment it applies. Not
//! all `DeploymentAction`s are valid for every `Deployment`, and neither are all `ClusterAction`
//! instances valid for every `Node`. For instance, we cannot delete a `Deployment` that does not
//! exist, nor can we increment/decrement the replicas of a `Deployment` that is not active.
//!
//! [`TraceEvent`] represents the Kubernetes API call which corresponds to a `ClusterAction`.
//!
//! [`Edge`] stores both a `ClusterAction` and the corresponding `TraceEvent`.
//!
//! [`Trace`] is a sequence of [`TraceEvent`]s along with some additional metadata. A `Trace` is
//! read by SimKube to drive a simulation.
//!
//!
//! ## The graph
//!
//! The Kubernetes cluster state graph is represented as a [`ClusterGraph`]. Walks of this graph map
//! 1:1 to traces which can be read by SimKube.
//!
//! ### Parameters
//! - [`trace_length`](Cli::trace_length): we construct the graph so as to contain all walks of
//!   length `trace_length` starting from the initial `Node`.
//! - `starting_state`: The initial [`Node`] from which to start the graph construction. We
//!   presently use a `Node` with no active [`Deployment`]s.
//! - `candidate_deployments`: A map from unique deployment names to corresponding initial
//!   [`Deployment`] configurations which are added whenever a `CreateDeployment` action is
//!   performed. We generate candidate deployments as `dep-1`, `dep-2`, etc. according to the
//!   [`deployment_count`](Cli::deployment_count) argument.
//!
//! ### Construction
//! - Starting from an initial [`Node`] with no active deployments, perform a breadth-first search.
//! - For each node visited:
//!   - Construct every [`ClusterAction`] applicable to the current `Node`, filtering for only those
//!     which produce a valid next `Node`.
//!   - Construct an [`Edge`] from the current `Node` to the next valid `Node`, recording both the
//!     `ClusterAction` and the corresponding `TraceEvent`.
//!   - Continue to a depth of `trace_length - 1` actions, such that the graph contains all walks on
//!     `trace_length` nodes from the initial `Node`.
//!
//! ## Extracting traces from the graph
//!
//!
//! [`Trace`] instances are obtained from the graph by enumerating all walks of length
//! `trace_length` through the graph via a depth-first search, and extracting the [`TraceEvent`]
//! from each [`Edge`].
//!
//! The graph generation and trace extraction steps are separated for conceptual simplicity, and in
//! anticipation of stochastic methods for trace generation.

mod output;

use std::collections::{
    BTreeMap,
    HashMap,
    HashSet,
    VecDeque,
};
use std::fmt::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use k8s_openapi::api::apps::v1::{
    Deployment as K8sDeployment,
    DeploymentSpec,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use kube::api::{
    DynamicObject,
    ObjectMeta,
};
use petgraph::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use sk_core::jsonutils::{
    ordered_eq,
    ordered_hash,
};
use sk_core::k8s::GVK;
use sk_core::prelude::corev1::{
    PodSpec,
    PodTemplateSpec,
};
use sk_store::{
    TraceEvent,
    TraceStorable,
    TraceStore,
    TracerConfig,
    TrackedObjectConfig,
};

use crate::output::{
    display_walks_and_traces,
    export_graphviz,
    gen_trace_event,
};

const BASE_TS: i64 = 1_728_334_068;

const REPLICA_COUNT_CHANGE: i32 = 1;
const REPLICA_COUNT_MIN: i32 = 0;
const REPLICA_COUNT_MAX: i32 = i32::MAX;

const RESOURCE_SCALE_FACTOR: f64 = 2.0;
const RESOURCE_SCALE_MIN: i64 = 1;

const SCALE_ACTION_PROBABILITY: f64 = 0.8;
const CREATE_DELETE_ACTION_PROBABILITY: f64 = 0.1;


// the clap crate allows us to define a CLI interface using a struct and some #[attributes]
/// `sk-gen` is a CLI tool for generating synthetic trace data which is ingestible by SimKube.
///
/// If no trace/walk output is requested, the tool will only generate the graph, which runs
/// considerably faster for substantially high input values.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Trace length (>= 3, including start state).
    ///
    /// A graph is constructed so as to contain all `trace_length`-walks from the starting state,
    /// then we enumerate all such walks.
    #[arg(short = 'l', long, value_parser = clap::value_parser!(u64).range(3..))]
    trace_length: u64,

    /// Number of candidate deployments
    ///
    /// These are generated as `dep-1`, `dep-2`, ... `dep-N`.
    #[arg(short, long)]
    source: PathBuf,

    /// Number of sample walks to generate (if not specified, generates all possible walks)
    #[arg(short, long)]
    num_samples: Option<usize>,

    /// If provided, output file in which graphviz representation of the graph will be written.
    #[arg(short = 'g', long)]
    graph_output_file: Option<PathBuf>,

    /// If provided, output directory to which traces will be written.
    ///
    /// Traces are stored as msgpack files of the form `trace-{n}.mp`. Each can be read individually
    /// by SimKube.
    #[arg(short = 'o', long)]
    traces_output_dir: Option<PathBuf>,

    /// Display walks to stdout. Walks are displayed as a list of nodes and intermediate actions.
    #[arg(short = 'w', long)]
    display_walks: bool,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum ObjectAction {
    Create,
    Delete,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum ActionType {
    Increase,
    Decrease,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum ResourceAction {
    Request { resource: String, action: ActionType },
    Limit { resource: String, action: ActionType },
    Claim,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum ContainerAction {
    Resource(ResourceAction),
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum DeploymentAction {
    ReplicaCount(ActionType),
    Object(ObjectAction),
    Container { name: String, action: ContainerAction },
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct ClusterAction {
    target_name: String,
    action_type: DeploymentAction,
}

#[derive(Clone, PartialEq, Debug)]
struct Deployment {
    deployment: K8sDeployment,
}

#[derive(Clone, Debug)]
struct Node {
    objects: BTreeMap<String, DynamicObject>,
}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        ordered_hash(&serde_json::to_value(&self.objects).unwrap()).hash(state);
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        ordered_eq(&serde_json::to_value(&self.objects).unwrap(), &serde_json::to_value(&other.objects).unwrap())
    }
}

impl Eq for Node {}

fn dynamic_object_to_deployment(dynamic_object: &DynamicObject) -> Result<K8sDeployment> {
    let json = serde_json::to_value(dynamic_object).expect("All dynamic objects are serializable");
    // TODO: check explicitly that this is a deployment
    let deployment = serde_json::from_value(json)?;
    Ok(deployment)
}

fn deployment_to_dynamic_object(deployment: &K8sDeployment) -> Result<DynamicObject> {
    let json = serde_json::to_value(deployment).expect("All deployments are serializable");
    let dynamic_object = serde_json::from_value(json).expect("DynamicObject should superset Deployment");
    Ok(dynamic_object)
}

fn scale_quantity(quantity_str: &str, scale: f64) -> Option<String> {
    // Parse number and suffix (e.g., "1048576" -> (1048576, ""))
    let mut num_str = String::new();
    let mut suffix = String::new();
    let mut in_suffix = false;

    for c in quantity_str.chars() {
        if c.is_ascii_digit() || c == '.' {
            if !in_suffix {
                num_str.push(c);
            } else {
                return None; // Invalid format
            }
        } else {
            in_suffix = true;
            suffix.push(c);
        }
    }

    // Parse the number
    let num: f64 = num_str.parse().ok()?;
    let scaled = (num * scale) as i64;

    if scaled < RESOURCE_SCALE_MIN {
        return None;
    }

    // Format back with the same suffix
    Some(format!("{}{}", scaled, suffix))
}

impl Node {
    fn new() -> Self {
        Self { objects: BTreeMap::new() }
    }

    fn from_trace_store(trace_store: &TraceStore) -> (Vec<Self>, BTreeMap<String, DynamicObject>) {
        let mut candidate_objects = BTreeMap::new();

        let mut node = Node::new();
        let mut nodes = Vec::new();

        for (event, _ts) in trace_store.iter() {
            // TODO: add ts handling

            for applied_obj in &event.applied_objs {
                let name = applied_obj.metadata.name.as_ref().unwrap();
                node.objects.insert(name.clone(), applied_obj.clone());

                if !candidate_objects.contains_key(name) {
                    candidate_objects.insert(name.clone(), applied_obj.clone());
                }
            }

            for deleted_obj in &event.deleted_objs {
                node.objects.remove(deleted_obj.metadata.name.as_ref().unwrap());
            }

            nodes.push(node.clone());
        }
        (nodes, candidate_objects)
    }

    fn create_deployment(&self, name: &str, candidate_deployments: &BTreeMap<String, DynamicObject>) -> Option<Self> {
        let deployment = candidate_deployments.get(name)?;

        let mut next_state = self.clone();
        next_state.objects.insert(name.to_string(), deployment.clone());
        Some(next_state)
    }

    fn delete_deployment(&self, name: &str) -> Option<Self> {
        if self.objects.contains_key(name) {
            let mut next_state = self.clone();
            next_state.objects.remove(name);
            Some(next_state)
        } else {
            None
        }
    }

    fn change_replica_count(&self, name: String, change: i32) -> Option<Self> {
        let mut deployment = dynamic_object_to_deployment(self.objects.get(&name)?).ok()?;

        let replicas = deployment.spec.get_or_insert_with(Default::default).replicas.get_or_insert(1);
        *replicas = replicas.checked_add(change)?;

        if *replicas < REPLICA_COUNT_MIN || *replicas > REPLICA_COUNT_MAX {
            return None;
        }

        let incremented_deployment = deployment_to_dynamic_object(&deployment).unwrap();

        let mut next_state = self.clone();
        next_state.objects.insert(name, incremented_deployment);
        Some(next_state)
    }

    fn resource_request(
        &self,
        deployment_name: String,
        container_name: String,
        action: ResourceAction,
    ) -> Option<Self> {
        let mut deployment = dynamic_object_to_deployment(self.objects.get(&deployment_name)?).ok()?;

        let resources = deployment
            .spec
            .get_or_insert_with(Default::default)
            .template
            .spec
            .get_or_insert_with(Default::default)
            .containers
            .iter_mut()
            .find(|container| container.name == container_name)?
            .resources
            .get_or_insert_with(Default::default);

        match action {
            ResourceAction::Request { resource, action } => {
                let requests = resources.requests.get_or_insert_with(BTreeMap::new);
                if let Some(current) = requests.get(&resource) {
                    let scale = match action {
                        ActionType::Increase => RESOURCE_SCALE_FACTOR,
                        ActionType::Decrease => 1.0 / RESOURCE_SCALE_FACTOR,
                    };
                    let new_value = scale_quantity(&current.0, scale)?;
                    requests.insert(resource, Quantity(new_value));
                }
            },
            ResourceAction::Limit { .. } => todo!(),
            ResourceAction::Claim => todo!(),
        }

        let updated_deployment = deployment_to_dynamic_object(&deployment).ok()?;
        let mut next_state = self.clone();
        next_state.objects.insert(deployment_name, updated_deployment);
        Some(next_state)
    }

    fn perform_action(
        &self,
        ClusterAction { target_name: deployment_name, action_type }: ClusterAction,
        candidate_deployments: &BTreeMap<String, DynamicObject>,
    ) -> Option<Self> {
        match action_type {
            DeploymentAction::ReplicaCount(ActionType::Increase) => {
                self.change_replica_count(deployment_name, REPLICA_COUNT_CHANGE)
            },
            DeploymentAction::ReplicaCount(ActionType::Decrease) => {
                self.change_replica_count(deployment_name, -REPLICA_COUNT_CHANGE)
            },
            DeploymentAction::Object(ObjectAction::Create) => {
                self.create_deployment(&deployment_name, candidate_deployments)
            },
            DeploymentAction::Object(ObjectAction::Delete) => self.delete_deployment(&deployment_name),
            DeploymentAction::Container { name: container_name, action } => match action {
                ContainerAction::Resource(resource_action) => {
                    self.resource_request(deployment_name, container_name, resource_action)
                },
            },
        }
    }

    fn deployments(&self) -> impl Iterator<Item = Deployment> + '_ {
        self.objects.values().filter_map(|obj| {
            dynamic_object_to_deployment(obj)
                .ok()
                .map(|deployment| Deployment { deployment })
        })
    }

    fn enumerate_actions(&self, candidate_deployments: &BTreeMap<String, DynamicObject>) -> Vec<ClusterAction> {
        let mut actions = Vec::new();

        // across all candidate deployments, we can try to create/delete according to whether the deployment
        // is already present
        for name in candidate_deployments.keys() {
            if self.objects.contains_key(name) {
                // already created, so we can delete
                actions.push(ClusterAction {
                    target_name: name.clone(),
                    action_type: DeploymentAction::Object(ObjectAction::Delete),
                });
            } else {
                // not already created, so we can create
                actions.push(ClusterAction {
                    target_name: name.clone(),
                    action_type: DeploymentAction::Object(ObjectAction::Create),
                });
            }
        }

        // across all active deployments, we can try to increment/decrement, saving bounds checks for later
        for Deployment { deployment } in self.deployments() {
            let Some(target_name) = deployment.metadata.name.clone() else {
                continue;
            };

            actions.push(ClusterAction {
                target_name: target_name.clone(),
                action_type: DeploymentAction::ReplicaCount(ActionType::Increase),
            });
            actions.push(ClusterAction {
                target_name: target_name.clone(),
                action_type: DeploymentAction::ReplicaCount(ActionType::Decrease),
            });

            let containers = deployment
                .spec
                .and_then(|spec| spec.template.spec)
                .map(|template| template.containers)
                .unwrap_or_default();

            for container in containers {
                let name = container.name;
                for action in [ActionType::Increase, ActionType::Decrease] {
                    for resource in ["memory", "cpu"] {
                        actions.push(ClusterAction {
                            target_name: target_name.clone(),
                            action_type: DeploymentAction::Container {
                                name: name.clone(),
                                action: ContainerAction::Resource(ResourceAction::Request {
                                    resource: resource.to_string(),
                                    action: action.clone(),
                                }),
                            },
                        });
                    }
                }
            }
        }

        actions
    }

    fn valid_action_states(&self, candidate_objects: &BTreeMap<String, DynamicObject>) -> Vec<(ClusterAction, Self)> {
        let actions = self.enumerate_actions(candidate_objects)
            .into_iter()
            .filter_map(|action| {
                self.perform_action(action.clone(), candidate_objects)
                    .map(|next_state| (action, next_state))
            })
            .collect::<Vec<_>>();

        dbg!(&actions.len());
        actions
    }
}

#[derive(Debug, Clone)]
struct Edge {
    action: ClusterAction,
    trace_event: TraceEvent,
}

type Walk = Vec<(Option<Edge>, Node)>;

struct ClusterGraph {
    candidate_objects: BTreeMap<String, DynamicObject>,
    graph: DiGraph<Node, Edge>,
}

impl ClusterGraph {
    fn new(candidate_objects: BTreeMap<String, DynamicObject>, starting_state: Vec<Node>, trace_length: u64) -> Self {
        let mut cluster_graph = Self { candidate_objects, graph: DiGraph::new() };

        // we want to track nodes we've seen before to prevent duplicates...
        // petgraph may have internal capabilities for this, but I haven't had the time to look
        // if this stays a part of our code, we may want to wrap the graph w/ tracking data in a new struct
        // -HM
        let mut node_to_index: HashMap<Node, NodeIndex> = HashMap::new();
        for node in &starting_state {
            let node_idx = cluster_graph.graph.add_node(node.clone());
            node_to_index.insert(node.clone(), node_idx);
        }

        // To find the graph containing all valid traces of trace_length with a given start state, we
        // perform bfs to a depth of trace_length. Queue item: `(depth, deployment)`
        let mut bfs_queue: VecDeque<(u64, Node)> = VecDeque::new();
        for node in starting_state {
            bfs_queue.push_back((1, node));
        }
        let mut visited = HashSet::new();

        while let Some((depth, node)) = bfs_queue.pop_front() {
            dbg!(&bfs_queue.len());
            let node_idx = *node_to_index.get(&node).expect("node not found in node_to_index");

            if depth >= trace_length {
                continue;
            }

            let not_previously_seen = visited.insert(node.clone());
            if !not_previously_seen {
                continue;
            }

            node.valid_action_states(&cluster_graph.candidate_objects)
                .into_iter()
                .for_each(|(action, next_state)| {
                    let next_idx = *node_to_index.entry(next_state.clone()).or_insert_with(|| {
                        let node = cluster_graph.graph.add_node(next_state.clone());
                        bfs_queue.push_back((depth + 1, next_state.clone()));
                        node
                    });

                    // We precompute the trace_event once here for our edge rather than recomputing it every
                    // time the edge is traversed in a walk.
                    let trace_event = gen_trace_event(BASE_TS + depth as i64, &node, &next_state);

                    // Because we are not revisiting outgoing nodes, we can be sure that the edge does not already exist
                    // so long as the same (node, node) edge is not achievable by distinct actions
                    cluster_graph
                        .graph
                        .update_edge(node_idx, next_idx, Edge { action, trace_event });
                });
        }

        cluster_graph
    }

    fn generate_walks(&self, trace_length: u64) -> Vec<Walk> {
        let start_nodes: Vec<NodeIndex> = self.graph.node_indices().take(1).collect();
        let mut all_walks = Vec::new();

        // We use a depth-first search because eventually we may want to use stochastic methods which do not
        // fully enumerate the neighborhood of each visited node.
        for walk_start_node in start_nodes {
            let walks = self.dfs_walks(walk_start_node, trace_length);
            all_walks.extend(walks.into_iter().map(|walk_indices| {
                let mut walk = Vec::new();

                let start_node = self.graph.node_weight(walk_indices[0]).unwrap().clone();
                walk.push((None, start_node));

                for window in walk_indices.windows(2) {
                    let (prev, next) = (window[0], window[1]);

                    let edge_idx = self.graph.find_edge(prev, next).unwrap();
                    let node = self.graph.node_weight(next).unwrap().clone();
                    let edge = self.graph.edge_weight(edge_idx).cloned().unwrap();
                    walk.push((Some(edge), node));
                }

                walk
            }));
        }

        all_walks
    }

    fn dfs_walks(&self, current_node: NodeIndex, walk_length: u64) -> Vec<Vec<NodeIndex>> {
        let mut walks = Vec::new();

        let start_walk = vec![current_node];
        self.dfs_walks_helper(current_node, start_walk, walk_length, &mut walks);

        walks
    }

    fn dfs_walks_helper(
        &self,
        current_node: NodeIndex,
        current_walk: Vec<NodeIndex>,
        walk_length: u64,
        walks: &mut Vec<Vec<NodeIndex>>,
    ) {
        if current_walk.len() as u64 == walk_length {
            walks.push(current_walk);
            return;
        }

        for neighbor in self.graph.neighbors(current_node) {
            let mut new_walk = current_walk.clone();
            new_walk.push(neighbor);
            self.dfs_walks_helper(neighbor, new_walk, walk_length, walks);
        }
    }

    fn to_graphviz(&self) -> String {
        let mut dot = String::new();
        writeln!(&mut dot, "digraph ClusterGraph {{").unwrap();

        // certain visualization software seem not to like this annotation, so it is presently omitted.
        // writeln!(&mut dot, "  node [shape=box];").unwrap();

        for node_index in self.graph.node_indices() {
            let node = &self.graph[node_index];
            let label = node
                .deployments()
                .map(|dep| {
                    format!(
                        "{}: {}",
                        dep.deployment.metadata.name.unwrap(),
                        dep.deployment.spec.unwrap().replicas.unwrap()
                    )
                })
                .collect::<Vec<_>>()
                .join("\\n");
            writeln!(&mut dot, "  {} [label=\"{}\"];", node_index.index(), label).unwrap();
        }

        for edge in self.graph.edge_references() {
            let action = &edge.weight().action;
            writeln!(
                &mut dot,
                "  {} -> {} [label=\"{} {}\"];",
                edge.source().index(),
                edge.target().index(),
                format!("{:?}", action),
                action.target_name.replace('"', "\\\"") // Escape any quotes in the name
            )
            .unwrap();
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    fn walks_with_sampling(&self, start_node: NodeIndex, walk_length: u64, num_samples: usize) -> Vec<Vec<NodeIndex>> {
        let mut rng = thread_rng();
        let mut samples = Vec::new();

        for _ in 0..num_samples {
            let mut current_walk = vec![start_node];
            let mut current_node = start_node;

            for _ in 1..walk_length {
                let neighbors: Vec<_> = self.graph.neighbors(current_node).collect();
                if neighbors.is_empty() {
                    break;
                }

                let weights: Vec<f64> = neighbors
                    .iter()
                    .map(|&n| {
                        let edge = self.graph.edge_weight(self.graph.find_edge(current_node, n).unwrap()).unwrap();
                        match edge.action.action_type {
                            DeploymentAction::ReplicaCount(_) | DeploymentAction::Container { .. } => {
                                SCALE_ACTION_PROBABILITY
                            },
                            DeploymentAction::Object(_) => CREATE_DELETE_ACTION_PROBABILITY,
                        }
                    })
                    .collect();

                let dist = WeightedIndex::new(&weights).unwrap();
                let next_node = neighbors[dist.sample(&mut rng)];

                current_walk.push(next_node);
                current_node = next_node;
            }

            samples.push(current_walk);
        }

        samples
    }

    fn generate_n_walks_with_sampling(&self, trace_length: u64, num_samples: usize) -> Vec<Walk> {
        let walk_start_node = self.graph.node_indices().next().unwrap();
        let sampled_walks = self.walks_with_sampling(walk_start_node, trace_length, num_samples);

        sampled_walks
            .into_iter()
            .map(|walk_indices| {
                let mut walk = Vec::new();

                let start_node = self.graph.node_weight(walk_indices[0]).unwrap().clone();
                walk.push((None, start_node));

                for window in walk_indices.windows(2) {
                    let (prev, next) = (window[0], window[1]);

                    let edge_idx = self.graph.find_edge(prev, next).unwrap();
                    let node = self.graph.node_weight(next).unwrap().clone();
                    let edge = self.graph.edge_weight(edge_idx).cloned().unwrap();
                    walk.push((Some(edge), node));
                }

                walk
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let input_trace_data: Vec<u8> = std::fs::read(&cli.source)?;

    let trace = TraceStore::import(input_trace_data, &None)?;

    let (nodes, candidate_deployments) = Node::from_trace_store(&trace);
    // TODO, we should use the trace as the start point wherever we start from

    println!("candidate_deployments:\n\n{:#?}\n\n\n nodes:\n\n{:#?}", candidate_deployments, nodes);
    // write to file
    std::fs::write("out/candidate_deployments.json", serde_json::to_string_pretty(&candidate_deployments)?).unwrap();
    for (i, node) in nodes.iter().enumerate() {
        std::fs::write(format!("out/node-{i}.json"), format!("{:#?}", node)).unwrap();
    }


    // Construct the graph by searching all valid sequences of `trace_length`-1 actions from the
    // starting state for a total of `trace_length` nodes.
    let starting_state = vec![nodes[nodes.len() - 1].clone()];
    // let starting_state = nodes[..1].to_vec();
    let graph = ClusterGraph::new(candidate_deployments, starting_state, cli.trace_length);

    // if the user provided a path for us to save the graphviz representation, do so
    if let Some(graph_output_file) = &cli.graph_output_file {
        export_graphviz(&graph, graph_output_file)?;
    }

    // If we don't need to output walks or traces, we don't need to generate them.
    if cli.graph_output_file.is_some() || cli.traces_output_dir.is_some() || cli.display_walks {
        let walks = if let Some(num_samples) = cli.num_samples {
            graph.generate_n_walks_with_sampling(cli.trace_length, num_samples)
        } else {
            graph.generate_walks(cli.trace_length)
        };

        let traces: Vec<TraceStore> = walks.iter().map(tracestore_from_walk).collect();

        display_walks_and_traces(&walks, &traces, &cli)?;
    }

    Ok(())
}

fn tracestore_from_walk(walk: &Walk) -> TraceStore {
    let config = TracerConfig {
        tracked_objects: HashMap::from([(
            GVK::new("apps", "v1", "Deployment"),
            TrackedObjectConfig {
                track_lifecycle: false,
                pod_spec_template_path: None,
            },
        )]),
    };

    let mut trace_store = TraceStore::new(config);

    let events = walk
        .iter()
        .filter_map(|(edge, _node)| edge.as_ref().map(|e| e.trace_event.clone()))
        .collect::<Vec<_>>();

    for (ts, trace_event) in events.into_iter().enumerate() {
        for obj in trace_event.applied_objs {
            trace_store.create_or_update_obj(&obj, ts as i64, None); // TODO check on maybe_old_hash
        }

        for obj in trace_event.deleted_objs {
            trace_store.delete_obj(&obj, ts as i64);
        }
    }

    trace_store
}
