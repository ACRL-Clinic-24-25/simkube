//! Utility functions for the trace generation stage.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use chrono::{
    DateTime,
    Utc,
};
use daft::Diffable;
use kube::api::DynamicObject;
use serde_json::json;
use sk_core::k8s::GVK;
use sk_store::{
    TraceEvent,
    TraceStorable,
    TraceStore,
    TracerConfig,
    TrackedObjectConfig,
};
use tracing::{
    debug,
    instrument,
    warn,
};

use crate::model::{
    DynamicObjectNewType,
    ObjectKey,
};

/// Convert a list of [`TraceEvent`]s into a [`TraceStore`] so that downstream tooling (e.g.
/// SimKube replay) can consume them.
#[instrument(skip(events), fields(event_count = events.len()))]
pub fn tracestore_from_events(events: &[TraceEvent]) -> TraceStore {
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

    for (ts, trace_event) in events.iter().enumerate() {
        for obj in trace_event.applied_objs.clone() {
            // ignore `maybe_old_hash` – we do not track lifecycle here
            #[allow(clippy::cast_possible_wrap)]
            trace_store.create_or_update_obj(&obj, ts as i64, None).unwrap();
        }
        for obj in trace_event.deleted_objs.clone() {
            #[allow(clippy::cast_possible_wrap)]
            trace_store.delete_obj(&obj, ts as i64).unwrap();
        }
    }

    trace_store
}

/// Compute the difference between two object maps, returning new/updated (applied) and removed
/// (deleted) objects.
#[must_use]
pub fn diff_objects<'a>(
    before: &'a std::collections::BTreeMap<ObjectKey, DynamicObjectNewType>,
    after: &'a std::collections::BTreeMap<ObjectKey, DynamicObjectNewType>,
) -> (Vec<DynamicObject>, Vec<DynamicObject>) {
    let diff = before.diff(after);

    // Newly added keys correspond to created objects – return the *new* value.
    let mut applied: Vec<DynamicObject> = diff.added.values().map(|v| v.dynamic_object.clone()).collect();

    // Removed keys correspond to deleted objects – return the *old* value.
    let deleted: Vec<DynamicObject> = diff.removed.values().map(|v| v.dynamic_object.clone()).collect();

    // Modified keys: treat as update, take the *after* value.
    applied.extend(diff.modified_values().map(|leaf| leaf.after.dynamic_object.clone()));

    (applied, deleted)
}

/// Create a timestamped output directory under `runs/` and write basic metadata.
#[instrument]
pub fn create_timestamped_output_dir() -> Result<PathBuf> {
    let base_dir = PathBuf::from("runs");
    std::fs::create_dir_all(&base_dir)?;

    let now: DateTime<Utc> = SystemTime::now().into();
    let timestamp = now.to_rfc3339().replace([':', '.'], "-"); // make filesystem-friendly
    let output_dir = base_dir.join(timestamp);
    std::fs::create_dir_all(&output_dir)?;

    let metadata = json!({
        "timestamp": now.to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
        "command_args": std::env::args().collect::<Vec<_>>()
    });

    let metadata_path = output_dir.join("metadata.json");
    let mut file = File::create(&metadata_path)?;
    file.write_all(serde_json::to_string_pretty(&metadata)?.as_bytes())?;

    Ok(output_dir)
}

/// Write DOT graph description to a file within `output_dir` and return the path.
#[instrument(skip(dot_content))]
pub fn write_dot_file(output_dir: &Path, filename: &str, dot_content: &str) -> Result<PathBuf> {
    let file_path = output_dir.join(filename);
    let mut file = File::create(&file_path)?;
    write!(file, "{dot_content}")?;

    debug!("Graph written to: {}", file_path.display());
    Ok(file_path)
}
