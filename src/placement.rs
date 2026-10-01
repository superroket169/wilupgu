use std::collections::HashMap;

use crate::device::DeviceId;
use crate::node::{NodeId, NodeSpec};

#[derive(Debug, Clone, Default)]
pub struct Placement {
    nodes: HashMap<NodeId, Vec<DeviceId>>,
}

impl Placement {
    pub fn new() -> Self {
        Self::default()
    }

    /// The only way to add a node
    pub fn assign(&mut self, node: &NodeSpec, devices: Vec<DeviceId>) -> Result<(), String> {
        let name = node.shader().name;
        if devices.is_empty() {
            return Err(format!(
                "node {:?} (`{name}`) is placed on no device",
                node.id()
            ));
        }
        for (i, d) in devices.iter().enumerate() {
            if devices[..i].contains(d) {
                return Err(format!(
                    "node {:?} (`{name}`) lists device {d:?} twice",
                    node.id()
                ));
            }
        }
        self.nodes.insert(node.id(), devices);
        Ok(())
    }

    pub fn devices(&self, node: NodeId) -> Option<&[DeviceId]> {
        self.nodes.get(&node).map(Vec::as_slice)
    }

    /// validates a whole node spec list
    /// do NOT add this to HashMap, just for verify it
    pub(crate) fn validate(&self, specs: &[NodeSpec]) -> Result<(), String> {
        for spec in specs {
            if !self.nodes.contains_key(&spec.id()) {
                return Err(format!(
                    "node {:?} (`{}`) has no placement",
                    spec.id(),
                    spec.shader().name
                ));
            }
        }
        for id in self.nodes.keys() {
            if !specs.iter().any(|s| s.id() == *id) {
                return Err(format!(
                    "placement names node {id:?}, which isn't in the spec list"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/placement.rs"]
mod tests;
