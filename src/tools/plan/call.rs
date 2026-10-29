//! [`CallSpec`]: one call of a node on its tensors, before it's built.
//! Built, it is a [`Call`]: the node, and the ids of the tensors it runs on.

use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{rules, ComputeDevice, Node};
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::node::{MetaSource, NodeSpec};
use crate::tools::plan::tensor::TensorSpec;

/// What a [`CallSpec`] is made from.
pub struct CallDesc {
    /// The node to call.
    pub node: NodeSpec,
    /// One tensor per slot of the node's shader, in slot order. The tensors
    /// belong to the plan; a call only names them, so several can share one.
    pub tensors: Vec<GlobalId<TensorSpec>>,
}

/// One call of a node on its tensors.
pub struct CallSpec {
    id: GlobalId<CallSpec>,
    desc: CallDesc,
}

impl CallSpec {
    /// The node to call.
    pub fn node(&self) -> &NodeSpec {
        &self.desc.node
    }

    /// The tensor of each slot, in slot order.
    pub fn tensors(&self) -> &[GlobalId<TensorSpec>] {
        &self.desc.tensors
    }

    /// Fails if there isn't one tensor per slot of the node's shader.
    pub fn new(desc: CallDesc) -> Result<Self, Error> {
        rules::slot_count(desc.node.shader().signature(), desc.tensors.len())?;
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    pub fn id(&self) -> GlobalId<CallSpec> {
        self.id
    }

    /// [`NodeSpec::ready`].
    pub fn ready(&self) -> Result<(), Error> {
        self.desc.node.ready()
    }

    /// Builds the node; the tensors are the plan's to build.
    pub fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Call<'d, B>, Error> {
        Ok(Call {
            node: self.desc.node.build(device)?,
            tensors: self.desc.tensors.clone(),
            sources: self.desc.node.meta().to_vec(),
        })
    }
}

/// A built [`CallSpec`]: the node, the tensor of each of its slots, and
/// where each field of the node's meta comes from.
pub struct Call<'d, B: Backend + Topology> {
    pub(super) node: Node<'d, B>,
    pub(super) tensors: Vec<GlobalId<TensorSpec>>,
    pub(super) sources: Vec<MetaSource>,
}

impl<'d, B: Backend + Topology> Call<'d, B> {
    /// The node.
    pub fn node(&self) -> &Node<'d, B> {
        &self.node
    }

    /// The tensor of each slot, in slot order.
    pub fn tensors(&self) -> &[GlobalId<TensorSpec>] {
        &self.tensors
    }

    /// Where each field of the node's meta comes from.
    pub fn sources(&self) -> &[MetaSource] {
        &self.sources
    }
}

#[cfg(test)]
#[path = "../../tests/call.rs"]
mod tests;
