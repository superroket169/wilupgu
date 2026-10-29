//! [`NodeSpec`]: a node before it's built. It names the shader, where its
//! meta comes from, and how many workgroups it runs. Its tensors are given
//! by the [`CallSpec`](super::call::CallSpec) that calls it.

use crate::backend::shader::{MetaType, Shader, Workgroups};
use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{rules, ComputeDevice, MetaValue, Node};
use crate::tools::plan::deferred::{Dynamic, Resolvable};
use crate::tools::plan::id::GlobalId;

/// Where one meta field's value comes from.
#[derive(Debug, Clone)]
pub enum MetaSource {
    /// Known once, when the node is built.
    Uint(Resolvable<u32>),
    /// Known once, when the node is built.
    Float(Resolvable<f32>),
    /// Read from a [`Dynamic`] when the runnable's `apply_dynamics` is
    /// called, like an optimizer's step count.
    UintPerRun(Dynamic<u32>),
    /// Read from a [`Dynamic`] when the runnable's `apply_dynamics` is
    /// called, like a scheduled learning rate.
    FloatPerRun(Dynamic<f32>),
}

impl MetaSource {
    /// The type of field it fills.
    pub fn ty(&self) -> MetaType {
        match self {
            MetaSource::Uint(_) | MetaSource::UintPerRun(_) => MetaType::Uint,
            MetaSource::Float(_) | MetaSource::FloatPerRun(_) => MetaType::Float,
        }
    }

    /// Whether it is read from a [`Dynamic`].
    pub fn is_per_run(&self) -> bool {
        matches!(self, MetaSource::UintPerRun(_) | MetaSource::FloatPerRun(_))
    }

    // The value a node is built with; a per-run value is 0 until `apply_dynamics` sets it.
    pub(crate) fn build_value(&self) -> Result<MetaValue, Error> {
        Ok(match self {
            MetaSource::Uint(r) => MetaValue::Uint(*r.value()?),
            MetaSource::Float(r) => MetaValue::Float(*r.value()?),
            MetaSource::UintPerRun(_) => MetaValue::Uint(0),
            MetaSource::FloatPerRun(_) => MetaValue::Float(0.0),
        })
    }
}

/// What a [`NodeSpec`] is made from.
#[derive(Clone)]
pub struct NodeDesc {
    /// The shader the node runs.
    pub shader: Shader,
    /// One per field of the shader's meta, in order.
    pub meta: Vec<MetaSource>,
    /// How many workgroups one dispatch runs.
    pub workgroups: Resolvable<Workgroups>,
}

/// A node to build: a [`NodeDesc`], checked against its shader.
pub struct NodeSpec {
    id: GlobalId<NodeSpec>,
    desc: NodeDesc,
}

impl NodeSpec {
    /// The shader the node runs.
    pub fn shader(&self) -> &Shader {
        &self.desc.shader
    }

    /// Where each meta field comes from, in the shader's order.
    pub fn meta(&self) -> &[MetaSource] {
        &self.desc.meta
    }

    /// How many workgroups one dispatch runs.
    pub fn workgroups(&self) -> &Resolvable<Workgroups> {
        &self.desc.workgroups
    }

    /// Fails if the meta doesn't match the shader's fields in count and type.
    pub fn new(desc: NodeDesc) -> Result<Self, Error> {
        rules::meta_types(
            desc.shader.signature(),
            desc.meta.iter().map(MetaSource::ty),
        )?;
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    pub fn id(&self) -> GlobalId<NodeSpec> {
        self.id
    }

    /// Fails while the workgroup count or a value known once isn't known.
    pub fn ready(&self) -> Result<(), Error> {
        self.desc.workgroups.value()?;
        for source in &self.desc.meta {
            source.build_value()?;
        }
        Ok(())
    }

    /// Builds the node with every value known once; a per-run value is 0
    /// until the runnable's `apply_dynamics` sets it.
    pub fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Node<'d, B>, Error> {
        let meta = self
            .desc
            .meta
            .iter()
            .map(MetaSource::build_value)
            .collect::<Result<Vec<_>, _>>()?;
        device.node(&self.desc.shader, &meta, *self.desc.workgroups.value()?)
    }
}

#[cfg(test)]
#[path = "../../tests/node.rs"]
mod tests;
