//! [`RunPlan`]: tensors and nodes to build together, and the order their
//! nodes run in. Built, it is a [`Runnable`].

use std::collections::HashMap;

use crate::backend::dtype::DataKind;
use crate::backend::shader::BindingRole;
use crate::backend::{Backend, Topology};
use crate::safe::ComputeDevice;
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::node::{NodeId, NodeSpec};
use crate::tools::plan::runnable::Runnable;
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::{TensorId, TensorSpec};

/// What a [`RunPlan`] is made from.
pub struct PlanDesc {
    /// Every tensor a node of the plan binds.
    pub tensors: Vec<TensorSpec>,
    /// The nodes to build and run.
    pub nodes: Vec<NodeSpec>,
    /// The order the nodes run in. Every node is in it, at least once.
    pub order: Vec<NodeId>,
}

/// Tensors and nodes to build on one device, and the order the nodes run in.
pub struct RunPlan {
    id: GlobalId<RunPlan>,
    desc: PlanDesc,
}

impl RunPlan {
    /// The tensors to build.
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.desc.tensors
    }

    /// The nodes to build.
    pub fn nodes(&self) -> &[NodeSpec] {
        &self.desc.nodes
    }

    /// The order the nodes run in.
    pub fn order(&self) -> &[NodeId] {
        &self.desc.order
    }
}

impl Spec for RunPlan {
    type Description = PlanDesc;
    type Outcome<'d, B: Backend + Topology + 'd> = Runnable<'d, B>;

    /// Fails if a node binds a tensor that isn't in the plan, or one of
    /// another kind than its slot's; if the order names a node that isn't in
    /// the plan; or if a node isn't in the order.
    fn new(desc: PlanDesc) -> Result<Self, String> {
        let kinds: HashMap<TensorId, _> = desc.tensors.iter().map(|t| (t.id(), t.kind())).collect();
        for node in &desc.nodes {
            let shader = node.shader();

            for (binding, role) in node.bindings().iter().zip(shader.layout) {
                match kinds.get(&binding.tensor) {
                    None => {
                        return Err(format!(
                            "shader `{}`: slot {} binds a tensor that isn't in the plan",
                            shader.name, binding.slot
                        ))
                    }
                    Some(got) if *got != role_kind(role) => {
                        return Err(format!(
                            "shader `{}`: slot {} is {:?}, its tensor is {got:?}",
                            shader.name,
                            binding.slot,
                            role_kind(role)
                        ))
                    }
                    Some(_) => {}
                }
            }
        }
        let mut in_order: HashMap<NodeId, bool> =
            desc.nodes.iter().map(|n| (n.id(), false)).collect();

        for id in &desc.order {
            *in_order
                .get_mut(id)
                .ok_or("the order names a node that isn't in the plan")? = true;
        }

        if let Some(node) = desc.nodes.iter().find(|n| !in_order[&n.id()]) {
            return Err(format!(
                "a node of shader `{}` isn't in the order",
                node.shader().name
            ));
        }

        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    fn id(&self) -> GlobalId<RunPlan> {
        self.id
    }

    /// Fails where a tensor's or node's [`Spec::ready`] does.
    fn ready(&self) -> Result<(), String> {
        for tensor in &self.desc.tensors {
            tensor.ready()?;
        }
        for node in &self.desc.nodes {
            node.ready()?;
        }
        Ok(())
    }

    /// Builds every tensor, then every node.
    fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Runnable<'d, B>, String> {
        self.ready()?;
        Runnable::build(self, device)
    }
}

pub(crate) fn role_kind(role: &BindingRole) -> DataKind {
    let (BindingRole::Input(kind)
    | BindingRole::Output(kind)
    | BindingRole::InOut(kind)
    | BindingRole::Accumulate(kind)) = role;
    *kind
}

#[cfg(test)]
#[path = "../../tests/run.rs"]
mod tests;
