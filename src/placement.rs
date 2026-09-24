use std::collections::HashMap;

use crate::id::GlobalId;
use crate::mesh::Parallelity;
use crate::traits::{DeviceId, NodeSpec};

#[derive(Debug, Clone, Default)]
pub struct Placement {
    pub nodes: HashMap<GlobalId, Vec<DeviceId>>,
}

impl Placement {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assign(&mut self, node: GlobalId, devices: Vec<DeviceId>) {
        self.nodes.insert(node, devices);
    }

    pub fn devices(&self, node: GlobalId) -> Option<&[DeviceId]> {
        self.nodes.get(&node).map(Vec::as_slice)
    }

    pub fn validate<Buf>(&self, specs: &[NodeSpec<Buf>]) -> Result<(), String> {
        for spec in specs {
            let devices = self.devices(spec.id).ok_or_else(|| {
                format!(
                    "node {:?} (`{}`) has no placement",
                    spec.id, spec.shader.name
                )
            })?;

            if devices.is_empty() {
                return Err(format!(
                    "node {:?} (`{}`) is placed on no device",
                    spec.id, spec.shader.name
                ));
            }
            for (i, d) in devices.iter().enumerate() {
                if devices[..i].contains(d) {
                    return Err(format!(
                        "node {:?} (`{}`) lists device {d:?} twice",
                        spec.id, spec.shader.name
                    ));
                }
            }
            if spec.parallelity == Parallelity::Pipeline && devices.len() != 1 {
                return Err(format!(
                    "node {:?} (`{}`) is Pipeline but placed on {} devices, needs exactly 1",
                    spec.id,
                    spec.shader.name,
                    devices.len()
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
        p.assign(specs[0].id, vec![DeviceId(0), DeviceId(1)]);
        p.assign(specs[1].id, vec![DeviceId(1)]);
        assert!(p.validate(&specs).is_ok());
    }

    #[test]
    fn rejects_unplaced_node() {
        let specs = [spec(Parallelity::Data)];
        let err = Placement::new().validate(&specs).unwrap_err();
        assert!(err.contains("has no placement"), "{err}");
    }

    #[test]
    fn rejects_pipeline_on_two_devices() {
        let specs = [spec(Parallelity::Pipeline)];
        let mut p = Placement::new();
        p.assign(specs[0].id, vec![DeviceId(0), DeviceId(1)]);
        let err = p.validate(&specs).unwrap_err();
        assert!(err.contains("needs exactly 1"), "{err}");
    }

    #[test]
    fn rejects_duplicate_device() {
        let specs = [spec(Parallelity::Tensor)];
        let mut p = Placement::new();
        p.assign(specs[0].id, vec![DeviceId(0), DeviceId(0)]);
        let err = p.validate(&specs).unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn rejects_unknown_node() {
        let specs = [spec(Parallelity::Data)];
        let mut p = Placement::new();
        p.assign(specs[0].id, vec![DeviceId(0)]);
        p.assign(GlobalId::new(), vec![DeviceId(0)]);
        let err = p.validate(&specs).unwrap_err();
        assert!(err.contains("isn't in the spec list"), "{err}");
    }
}
