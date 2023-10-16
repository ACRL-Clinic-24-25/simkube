use k8s_openapi::apimachinery::pkg::apis::meta::v1 as metav1;
use kube::api::DynamicObject;
use serde::Deserialize;

use super::TraceEvent;
use crate::k8s::KubeResourceExt;

// TODO: we also want to add "positive" selectors, e.g., only take objects in a particular
// namespace or matching a particular label-selector.
#[derive(Default, Deserialize, Debug, Clone)]
pub struct TraceFilter {
    pub excluded_namespaces: Vec<String>,
    pub excluded_labels: Vec<metav1::LabelSelector>,
    pub exclude_daemonsets: bool,
}

pub fn filter_event(evt: &TraceEvent, f: &TraceFilter) -> Option<TraceEvent> {
    let new_evt = TraceEvent {
        ts: evt.ts,
        applied_objs: evt
            .applied_objs
            .iter()
            .filter(|obj| !obj_matches_filter(obj, f))
            .cloned()
            .collect(),
        deleted_objs: evt
            .deleted_objs
            .iter()
            .filter(|obj| !obj_matches_filter(obj, f))
            .cloned()
            .collect(),
    };

    if new_evt.applied_objs.is_empty() && new_evt.deleted_objs.is_empty() {
        return None;
    }

    Some(new_evt)
}

fn obj_matches_filter(obj: &DynamicObject, f: &TraceFilter) -> bool {
    obj.metadata
        .namespace
        .as_ref()
        .is_some_and(|ns| f.excluded_namespaces.contains(ns))
        || obj
            .metadata
            .owner_references
            .as_ref()
            .is_some_and(|owners| owners.iter().any(|owner| &owner.kind == "DaemonSet"))
        // TODO: maybe don't call unwrap here?  Right now we panic if the user specifies
        // an invalid label selector.  Or, maybe it doesn't matter once we write the CLI
        // tool.
        || f.excluded_labels.iter().any(|sel| obj.matches(sel).unwrap())
}