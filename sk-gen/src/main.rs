#![deny(
    // This is overly strict, of course. The intent is somewhat of a "quality seal," less to fix everything, and more to force us to add inline allows, which are even more needlessly verbose, but give us a mechanism to say "we think this is okay, but you might want to take a second look here."
    clippy::nursery,
    clippy::pedantic,
    // These are also just for clinic purposes
    missing_docs,
    clippy::missing_docs_in_private_items,
)]
//! `SimKube` Synthetic Trace Generator Command Line Interface (main entrypoint for the work of the
//! ACRL 24'-25' Clinic team)
//!
//! This binary provides a command-line interface to the sk-gen library, allowing users
//! to generate synthetic Kubernetes traces based on recorded [`ExportedTraces`](`ExportTrace`)
//! serialized as either JSON or `MessagePack` and custom expansion actions defined in JQ scripts
//! See binary --help for more information

use std::collections::BTreeMap;
use std::fs;
use std::path::{
    Path,
    PathBuf,
};

use anyhow::Result;
use clap::Parser;
use jaq_core::load::{
    Arena,
    Loader,
};
use jaq_core::{
    load,
    Compiler,
    Ctx,
    RcIter,
};
use jaq_json::Val;
use kube::api::DynamicObject;
use ordered_float::OrderedFloat;
use sk_gen::{
    diff_objects,
    Action,
    DynamicObjectNewType,
    Node,
    ObjectKey,
};
use sk_store::{
    ExportedTrace,
    TraceEvent,
};
use tracing::{
    debug,
    error,
    info,
    instrument,
    warn,
};

/// sk-gen command-line interface to generate synthetic traces for simkube from recorded traces and
/// graph expansion action scripts
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Number of synthetic traces to generate.
    #[arg(short, long)]
    num_samples: usize,

    /// Maximum length (in events) of each generated trace.
    #[arg(short = 'l', long)]
    trace_length: u64,

    /// Number of breadth-first enumeration layers to explore when expanding the state-space.
    #[arg(short = 'e', long, default_value_t = 3)]
    enumeration_steps: u64,

    /// Paths to files containing `SimKube` `ExportedTraces` serialized as either JSON or
    /// `MessagePack`.
    #[arg(short = 't', long)]
    pub input_traces: Vec<PathBuf>,

    /// Logging verbosity level (`trace`, `debug`, `info`, `warn`, `error`).
    #[arg(short, long, default_value = "info")]
    verbosity: String,

    /// Fraction of nodes to contract during the graph-contraction stage (range 0.0–1.0).
    #[arg(long, default_value_t = 0.5, value_parser = parse_contraction_strength)]
    contraction_strength: f64,

    // TODO: consider a more consistent import mechanism between traces and scripts
    /// Directory containing JQ scripts to import (format: {name}.jq)
    #[arg(short = 's', long)]
    script_directory: Option<PathBuf>,
}

/// Custom parser for `contraction_strength` to enforce range [0.0, 1.0]
fn parse_contraction_strength(s: &str) -> Result<f64, String> {
    let val: f64 = s.parse().map_err(|_| format!("'{s}' isn't a valid float number"))?;
    if (0.0..=1.0).contains(&val) {
        Ok(val)
    } else {
        Err(format!("value must be between 0.0 and 1.0, got: {val}"))
    }
}

/// Reads JQ scripts from a directory into a (filename, script) vector.
fn read_scripts_from_dir(dir_path: &PathBuf) -> Result<Vec<(String, String)>> {
    let mut scripts = Vec::new();

    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(anyhow::anyhow!("Script directory does not exist or is not a directory: {}", dir_path.display()));
    }

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(ext) = path.extension() {
            if ext == "jq" {
                if let Some(stem) = path.file_stem() {
                    if let Some(name) = stem.to_str() {
                        let content = fs::read_to_string(&path)?;
                        scripts.push((name.to_string(), content));
                    }
                }
            }
        }
    }

    Ok(scripts)
}

/// Processes a JQ script result and converts it to object lists
fn process_jq_results(results: Vec<jaq_json::Val>) -> impl Iterator<Item = BTreeMap<ObjectKey, DynamicObjectNewType>> {
    results
        .into_iter()
        .flat_map(|val| match serde_json::from_value::<Vec<Vec<DynamicObject>>>(val.into()) {
            Ok(dynamic_object_list) => dynamic_object_list,
            Err(e) => {
                error!("Error deserializing jaq result: {}", e);
                Vec::new()
            },
        })
        .map(|dynamic_object_list| {
            dynamic_object_list
                .into_iter()
                .map(|obj| (ObjectKey::from(&obj), DynamicObjectNewType { dynamic_object: obj }))
                .collect::<BTreeMap<_, _>>()
        })
}

/// Executes a JQ script on the provided JSON value
fn execute_jq_script(action_message: &str, jq_script: &str, objects_json: &serde_json::Value) -> Vec<jaq_json::Val> {
    let program = load::File { code: jq_script, path: () };

    let arena = Arena::default();
    let loader = Loader::new(jaq_std::defs().chain(jaq_json::defs()));

    let modules = match loader.load(&arena, program) {
        Ok(modules) => modules,
        Err(err) => {
            error!("Failed to load jaq script '{}': {:?}", action_message, err);
            return Vec::new();
        },
    };

    let filter = match Compiler::default()
        .with_funs(jaq_std::funs().chain(jaq_json::funs()))
        .compile(modules)
    {
        Ok(filter) => filter,
        Err(err) => {
            warn!("Failed to compile jaq script '{}': {:?}", action_message, err);
            return Vec::new();
        },
    };

    let inputs = RcIter::new(core::iter::empty());
    let jq_output = filter.run((Ctx::new([], &inputs), Val::from(objects_json.clone())));

    let mut results = Vec::new();
    for result in jq_output {
        match result {
            Ok(val) => results.push(val),
            Err(e) => {
                error!("Error running jaq filter for '{}': {}", action_message, e);
            },
        }
    }

    results
}

/// Creates actions from transformed objects
fn create_actions_from_objects<I>(node: &Node, objects_list: I, message: Option<&str>) -> Vec<Action>
where
    I: Iterator<Item = BTreeMap<ObjectKey, DynamicObjectNewType>>,
{
    objects_list
        .map(|objects| {
            let (applied_objs, deleted_objs) = diff_objects(&node.objects, &objects);

            Action {
                trace_event_newtype: TraceEvent { ts: node.ts + 1, applied_objs, deleted_objs }.into(),
                probability: OrderedFloat(1.0), /* TODO: non-uniform probabilities -- was previously blocked on not
                                                 * knowing that we were going to do jq for actions */
                message: message.map(str::to_owned),
            }
        })
        .collect()
}

/// Generates possible Kubernetes resource transformation actions for a given node.
///
/// This function analyzes the Kubernetes resources in the current node state and
/// creates actions that modify resource configurations in semantically meaningful ways.
/// Each action represents operations like:
/// - Doubling/halving memory allocations
/// - Doubling/halving CPU allocations
/// - Increasing/decreasing replica counts
///
/// Implementation Details: Uses jq-style scripts to transform the JSON representation of Kubernetes
/// objects, filters out invalid transformations (e.g., memory allocations that would be too small),
/// and constructs patch-based actions with appropriate probability weights.
#[instrument(skip(node, scripts), fields(object_count = node.objects.len()))]
fn next_action_fn(node: &Node, scripts: &[(String, String)]) -> Vec<Action> {
    if scripts.is_empty() {
        debug!("No scripts provided, no actions will be generated");
        return Vec::new();
    }

    let objects_json =
        serde_json::to_value(node.objects.values().map(|d| d.dynamic_object.clone()).collect::<Vec<_>>())
            .expect("Failed to serialize objects to JSON");

    scripts
        .iter()
        .flat_map(|(action_message, jq_script)| {
            let message = Some(action_message.clone());
            debug!("message: {:?}", message);

            let results = execute_jq_script(action_message, jq_script, &objects_json);
            let objects_list = process_jq_results(results);
            create_actions_from_objects(node, objects_list, message.as_deref())
        })
        .collect()
}

/// Loads traces from the provided file paths
fn load_input_traces(input_trace_paths: &[PathBuf]) -> Result<Vec<Vec<TraceEvent>>> {
    let mut input_traces = Vec::new();
    for path in input_trace_paths {
        info!("Loading trace from {}", path.display());
        let file = std::fs::File::open(path)?;

        let exported_trace: ExportedTrace = serde_json::from_reader(&file).or_else(|_| rmp_serde::from_read(&file))?;

        let events = exported_trace.events();
        input_traces.push(events);
    }
    Ok(input_traces)
}

/// Reports the outcome of loading JQ scripts in a single place, keeping the caller simple.
fn report_script_loads(path: &Path, count: usize) {
    if count == 0 {
        warn!("No JQ scripts found in {}", path.display());
    } else {
        info!("Loaded {} JQ scripts", count);
    }
}

/// Loads JQ scripts from the provided directory
fn load_jq_scripts(script_directory: Option<&PathBuf>) -> Vec<(String, String)> {
    let Some(path) = script_directory else {
        warn!("No script import path provided. Will not generate any actions during enumeration.");
        return Vec::new();
    };

    info!("Loading JQ scripts from {}", path.display());

    match read_scripts_from_dir(path) {
        Ok(scripts) => {
            report_script_loads(path, scripts.len());
            scripts
        },
        Err(e) => {
            error!("Failed to load JQ scripts: {}", e);
            Vec::new()
        },
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Conform to crate-standard logging.
    sk_core::logging::setup(&args.verbosity);
    info!("Starting simulation with {} input traces", args.input_traces.len());

    let input_traces = load_input_traces(&args.input_traces)?;
    let imported_scripts = load_jq_scripts(args.script_directory.as_ref());

    sk_gen::simulation::run(
        move |node: &Node| next_action_fn(node, &imported_scripts),
        input_traces,
        args.num_samples,
        args.trace_length,
        args.enumeration_steps,
        args.contraction_strength,
    )
}
