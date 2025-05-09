//! Data models for representing cluster state and transitions.
// TODO: it may be prefereable to implement the desired hash/eq behavior on the top-level types
// rather than doing the existing newtype cascade
use std::cmp::Ordering;
use std::collections::{
    BTreeMap,
    HashSet,
};
use std::hash::{
    Hash,
    Hasher,
};

use anyhow::Result;
use daft::Diffable;
use kube::api::DynamicObject;
use kube::Resource;
use ordered_float::OrderedFloat;
use serde::{
    Deserialize,
    Serialize,
};
use sk_core::jsonutils::{
    ordered_eq,
    ordered_hash,
};
use sk_core::k8s::GVK;
use sk_store::TraceEvent;
use tracing::{
    instrument,
    warn,
};

use crate::contraction_hierarchies::Distance;

/// Indicates whether a node/edge originates from observed traces or was synthesised during
/// enumeration.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ObjectType {
    /// Objects that were synthesized during the expansion stage.
    Synthetic,
    /// Objects that were directly observed in collected traces.
    Observed,
}

/// Primary key for objects in the state – `(group, version, kind, name)`.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Serialize)]
pub struct ObjectKey {
    /// Name of the Kubernetes resource
    pub(crate) name: String,
    /// Group, version, kind identifier for the resource type
    pub(crate) gvk: GVK,
    // TODO: namespace support to be added later
}

impl From<&DynamicObject> for ObjectKey {
    fn from(value: &DynamicObject) -> Self {
        let gvk = GVK::from_dynamic_obj(value).expect("dynamic object missing GVK");
        let name = value.meta().name.clone().expect("dynamic object missing name");
        Self { name, gvk }
    }
}

impl Ord for ObjectKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gvk
            .group
            .cmp(&other.gvk.group)
            .then(self.gvk.version.cmp(&other.gvk.version))
            .then(self.gvk.kind.cmp(&other.gvk.kind))
            .then(self.name.cmp(&other.name))
    }
}

impl PartialOrd for ObjectKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Wrapper around `DynamicObject` to provide custom `Eq`, `Hash` (order-insensitive JSON) and
/// `Diffable` implementations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicObjectNewType {
    /// The underlying Kubernetes resource object.
    pub dynamic_object: DynamicObject,
}

impl From<DynamicObject> for DynamicObjectNewType {
    fn from(value: DynamicObject) -> Self {
        Self { dynamic_object: value }
    }
}

impl From<DynamicObjectNewType> for DynamicObject {
    fn from(value: DynamicObjectNewType) -> Self {
        value.dynamic_object
    }
}

impl PartialEq for DynamicObjectNewType {
    fn eq(&self, other: &Self) -> bool {
        ordered_eq(
            &serde_json::to_value(&self.dynamic_object).unwrap(),
            &serde_json::to_value(&other.dynamic_object).unwrap(),
        )
    }
}

impl Eq for DynamicObjectNewType {}

impl Hash for DynamicObjectNewType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ordered_hash(&serde_json::to_value(&self.dynamic_object).unwrap()).hash(state);
    }
}

impl Diffable for DynamicObjectNewType {
    type Diff<'daft> = daft::Leaf<&'daft Self>;

    fn diff<'daft>(&'daft self, other: &'daft Self) -> Self::Diff<'daft> {
        daft::Leaf { before: self, after: other }
    }
}

/// Cluster state (set of objects) at a specific logical timestamp.
#[derive(Clone, Debug, Serialize)]
pub struct Node {
    /// Whether this state was observed in real traces input to sk-gen or synthesized during the
    /// expansion stage.
    pub object_type: ObjectType,
    /// Collection of Kubernetes resources present in this state.
    pub objects: BTreeMap<ObjectKey, DynamicObjectNewType>,
    /// Logical timestamp of this state.
    pub ts: i64,
}

// TODO: this is a placeholder implementation of time which minimizes the size of the state space by
// simply ignoring time. We are confident that there is interesting stuff being ignored with this
// approach. Equality & hashing ignore `ts` to avoid inflating the graph with identical states at
// different times (time is encoded in edges).
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.object_type == other.object_type && self.objects == other.objects
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.object_type.hash(state);
        self.objects.hash(state);
    }
}

impl Node {
    /// Apply a `TraceEvent` patch to the current state, returning the resulting state.
    #[instrument(level = "debug", skip(self, patch), fields(patch_ts = patch.ts))]
    pub fn apply_patch(&self, patch: &TraceEvent) -> Result<Node> {
        if patch.ts < self.ts {
            anyhow::bail!("patch is earlier than node timestamp");
        }

        let mut next = self.clone();
        next.ts = patch.ts;

        assert_eq!(
            patch
                .applied_objs
                .iter()
                .map(ObjectKey::from)
                .chain(patch.deleted_objs.iter().map(ObjectKey::from))
                .collect::<HashSet<_>>()
                .len(),
            patch.applied_objs.len() + patch.deleted_objs.len(),
            "objects must appear only once across the union of applied and deleted objects in a TraceEvent",
        );

        // Handle additions/updates
        for obj in &patch.applied_objs {
            let key = ObjectKey::from(obj);
            if let Some(existing) = next.objects.get(&key) {
                // TODO: shallow merge does not accurately reflect the behavior of kubernetes apply
                let existing_json = serde_json::to_value(&existing.dynamic_object).unwrap();
                let new_json = serde_json::to_value(obj).unwrap();
                if let (serde_json::Value::Object(mut existing_map), serde_json::Value::Object(new_map)) =
                    (existing_json, new_json)
                {
                    for (k, v) in new_map {
                        existing_map.insert(k, v);
                    }
                    if let Ok(merged_obj) =
                        serde_json::from_value::<DynamicObject>(serde_json::Value::Object(existing_map))
                    {
                        next.objects.insert(key, DynamicObjectNewType { dynamic_object: merged_obj });
                    } else {
                        warn!("failed to merge object, replacing");
                        next.objects.insert(key, obj.clone().into());
                    }
                } else {
                    // fall back to replace
                    next.objects.insert(key, obj.clone().into());
                }
            } else {
                next.objects.insert(key, obj.clone().into());
            }
        }

        // Handle deletions
        for obj in &patch.deleted_objs {
            next.objects.remove(&ObjectKey::from(obj));
        }

        Ok(next)
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            objects: BTreeMap::new(),
            ts: 0,
            object_type: ObjectType::Synthetic,
        }
    }
}

/// New-type wrapper with order-insensitive hashing/equality for whole `TraceEvent`s.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceEventNewType {
    /// The underlying trace event.
    pub trace_event: TraceEvent,
}

impl From<TraceEvent> for TraceEventNewType {
    fn from(value: TraceEvent) -> Self {
        Self { trace_event: value }
    }
}

impl std::ops::Deref for TraceEventNewType {
    type Target = TraceEvent;

    fn deref(&self) -> &Self::Target {
        &self.trace_event
    }
}

impl Hash for TraceEventNewType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ordered_hash(&serde_json::to_value(&self.trace_event).unwrap()).hash(state);
    }
}

impl Eq for TraceEventNewType {}

impl PartialEq for TraceEventNewType {
    fn eq(&self, other: &Self) -> bool {
        ordered_eq(
            &serde_json::to_value(&self.trace_event).unwrap(),
            &serde_json::to_value(&other.trace_event).unwrap(),
        )
    }
}

/// High-level description of an edge (patch + auxiliary info).
#[derive(Clone, Debug, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub struct Action {
    /// The trace event that transforms one state into another.
    pub trace_event_newtype: TraceEventNewType,
    /// Optional human-readable description of the action.
    pub message: Option<String>,
    /// Probability of taking this action when sampling.
    pub probability: OrderedFloat<f64>,
}

/// Edge payload stored in the graph.
#[derive(Clone, Debug, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub struct Edge {
    /// Whether this edge was observed in real traces or synthesized.
    pub object_type: ObjectType,
    /// The action that transforms the source node into the target node.
    pub action: Action,
}

impl Distance for Edge {
    fn probability(&self) -> OrderedFloat<f64> {
        self.action.probability
    }
}
