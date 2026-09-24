use std::collections::HashMap;

use crate::id::GlobalId;
use crate::mesh::Parallelity;
use crate::traits::{DeviceId, NodeSpec};

#[derive(Debug, Clone, Default)]
pub struct Placement {
    nodes: HashMap<GlobalId, Vec<DeviceId>>,
}

impl Placement {
    pub fn new() -> Self {
        Self::default()
    }

    /// The only way to add a node
    pub fn assign<Buf>(
        &mut self,
        node: &NodeSpec<Buf>,
        devices: Vec<DeviceId>,
    ) -> Result<(), String> {
        let name = node.shader.name;
        if devices.is_empty() {
            return Err(format!(
                "node {:?} (`{name}`) is placed on no device",
                node.id
            ));
        }
        for (i, d) in devices.iter().enumerate() {
            if devices[..i].contains(d) {
                return Err(format!(
                    "node {:?} (`{name}`) lists device {d:?} twice",
                    node.id
                ));
            }
        }
        if node.parallelity == Parallelity::Pipeline && devices.len() != 1 {
            return Err(format!(
                "node {:?} (`{name}`) is Pipeline but placed on {} devices, needs exactly 1",
                node.id,
                devices.len()
            ));
        }
        self.nodes.insert(node.id, devices);
        Ok(())
    }

    pub fn devices(&self, node: GlobalId) -> Option<&[DeviceId]> {
        self.nodes.get(&node).map(Vec::as_slice)
    }

    /// validates a whole node spec list
    /// do NOT add this to HashMap, just for verify it
    pub(crate) fn validate<Buf>(&self, specs: &[NodeSpec<Buf>]) -> Result<(), String> {
        for spec in specs {
            if !self.nodes.contains_key(&spec.id) {
                return Err(format!(
                    "node {:?} (`{}`) has no placement",
                    spec.id, spec.shader.name
                ));
            }
        }
        for id in self.nodes.keys() {
            if !specs.iter().any(|s| s.id == *id) {
                return Err(format!(
                    "placement names node {id:?}, which isn't in the spec list"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Shader, Workgroups};

    static EMPTY: Shader = Shader {
        name: "Empty",
        layout: &[],
        dispatch: &[],
    };

    fn spec(parallelity: Parallelity) -> NodeSpec<'static, ()> {
        NodeSpec {
            id: GlobalId::new(),
            shader: &EMPTY,
            bindings: &[],
            workgroups: Workgroups::linear(1),
            parallelity,
        }
    }

    #[test]
    fn accepts_a_complete_placement() {
        let specs = [spec(Parallelity::Data), spec(Parallelity::Pipeline)];
        let mut p = Placement::new();
        p.assign(&specs[0], vec![DeviceId(0), DeviceId(1)]).unwrap();
        p.assign(&specs[1], vec![DeviceId(1)]).unwrap();
        assert!(p.validate(&specs).is_ok());
    }

    #[test]
    fn assign_rejects_no_device() {
        let err = Placement::new()
            .assign(&spec(Parallelity::Data), vec![])
            .unwrap_err();
        assert!(err.contains("no device"), "{err}");
    }

    #[test]
    fn assign_rejects_pipeline_on_two_devices() {
        let err = Placement::new()
            .assign(&spec(Parallelity::Pipeline), vec![DeviceId(0), DeviceId(1)])
            .unwrap_err();
        assert!(err.contains("needs exactly 1"), "{err}");
    }

    #[test]
    fn assign_rejects_duplicate_device() {
        let err = Placement::new()
            .assign(&spec(Parallelity::Tensor), vec![DeviceId(0), DeviceId(0)])
            .unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn rejected_assign_leaves_nothing_behind() {
        let s = spec(Parallelity::Pipeline);
        let mut p = Placement::new();
        let _ = p.assign(&s, vec![DeviceId(0), DeviceId(1)]);
        assert!(p.devices(s.id).is_none());
    }

    #[test]
    fn validate_rejects_unplaced_node() {
        let specs = [spec(Parallelity::Data)];
        let err = Placement::new().validate(&specs).unwrap_err();
        assert!(err.contains("has no placement"), "{err}");
    }

    #[test]
    fn validate_rejects_node_outside_the_list() {
        let specs = [spec(Parallelity::Data)];
        let stranger = spec(Parallelity::Data);
        let mut p = Placement::new();
        p.assign(&specs[0], vec![DeviceId(0)]).unwrap();
        p.assign(&stranger, vec![DeviceId(0)]).unwrap();
        let err = p.validate(&specs).unwrap_err();
        assert!(err.contains("isn't in the spec list"), "{err}");
    }
}
