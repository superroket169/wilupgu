//! [`NodeSpec`]: one dispatch before it's built. It names the shader, where
//! its meta comes from, which tensors go in which slots, and how many
//! workgroups it runs.

use crate::backend::shader::{MetaType, Shader, Workgroups};
use crate::backend::{Backend, Topology};
use crate::safe::{ComputeDevice, MetaValue, Node};
use crate::tools::plan::deferred::{Dynamic, Resolvable};
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::TensorId;

/// Names one node of a plan.
pub type NodeId = GlobalId<NodeSpec>;

/// One tensor in one slot of a node's shader. Its role is the slot's, in
/// the shader's layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// As in [`Shader::layout`]: tensors are slots `1..=layout.len()`.
    pub slot: u32,
    /// The tensor that goes in it.
    pub tensor: TensorId,
}

impl Binding {
    /// `tensor` in `slot`.
    pub fn new(slot: u32, tensor: TensorId) -> Self {
        Self { slot, tensor }
    }
}

/// Where one meta field's value comes from.
#[derive(Debug, Clone)]
pub enum MetaSource {
    /// Known once, when the node is built.
    Uint(Resolvable<u32>),
    /// Known once, when the node is built.
    Float(Resolvable<f32>),
    /// Set again before every run, like an optimizer's step count.
    UintPerRun(Dynamic<u32>),
    /// Set again before every run, like a scheduled learning rate.
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

    /// Whether it is set again before every run.
    pub fn is_per_run(&self) -> bool {
        matches!(self, MetaSource::UintPerRun(_) | MetaSource::FloatPerRun(_))
    }

    // The value a node is built with; a per-run value is 0 until the run sets it.
    fn build_value(&self) -> Result<MetaValue, String> {
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
    pub shader: &'static Shader,
    /// One per field of the shader's meta, in order.
    pub meta: Vec<MetaSource>,
    /// One per slot of the shader's layout, in any order.
    pub bindings: Vec<Binding>,
    /// How many workgroups one dispatch runs.
    pub workgroups: Resolvable<Workgroups>,
}

/// A node to build: a [`NodeDesc`], checked against its shader. The bindings
/// aren't part of the built [`Node`]; a plan's run hands it those tensors.
pub struct NodeSpec {
    id: NodeId,
    desc: NodeDesc,
}

impl NodeSpec {
    /// The shader the node runs.
    pub fn shader(&self) -> &'static Shader {
        self.desc.shader
    }

    /// Where each meta field comes from, in the shader's order.
    pub fn meta(&self) -> &[MetaSource] {
        &self.desc.meta
    }

    /// Sorted by slot.
    pub fn bindings(&self) -> &[Binding] {
        &self.desc.bindings
    }

    /// How many workgroups one dispatch runs.
    pub fn workgroups(&self) -> &Resolvable<Workgroups> {
        &self.desc.workgroups
    }
}

impl Spec for NodeSpec {
    type Description = NodeDesc;
    type Outcome<'d, B: Backend + Topology + 'd> = Node<'d, B>;

    /// Fails if the meta doesn't match the shader's fields in count and type,
    /// or the bindings don't fill every slot of its layout exactly once.
    fn new(mut desc: NodeDesc) -> Result<Self, String> {
        let shader = desc.shader;
        if desc.meta.len() != shader.meta.len() {
            return Err(format!(
                "shader `{}`: expected {} meta fields, got {}",
                shader.name,
                shader.meta.len(),
                desc.meta.len()
            ));
        }
        for (field, source) in shader.meta.iter().zip(&desc.meta) {
            if field.ty != source.ty() {
                return Err(format!(
                    "shader `{}`: meta field `{}` is {:?}, got a {:?} source",
                    shader.name,
                    field.name,
                    field.ty,
                    source.ty()
                ));
            }
        }
        desc.bindings.sort_by_key(|b| b.slot);
        let slots: Vec<u32> = desc.bindings.iter().map(|b| b.slot).collect();
        let expected: Vec<u32> = (1..=shader.layout.len() as u32).collect();
        if slots != expected {
            return Err(format!(
                "shader `{}`: bindings must fill slots {expected:?} once each, got {slots:?}",
                shader.name
            ));
        }
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    fn id(&self) -> NodeId {
        self.id
    }

    /// Fails while the workgroup count or a value known once isn't known.
    fn ready(&self) -> Result<(), String> {
        self.desc.workgroups.value()?;
        for source in &self.desc.meta {
            source.build_value()?;
        }
        Ok(())
    }

    /// Builds the node with every value known once; a per-run value is 0
    /// until a run sets it.
    fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Node<'d, B>, String> {
        let meta = self
            .desc
            .meta
            .iter()
            .map(MetaSource::build_value)
            .collect::<Result<Vec<_>, _>>()?;
        device.node(self.desc.shader, &meta, *self.desc.workgroups.value()?)
    }
}

#[cfg(test)]
#[path = "../../tests/node.rs"]
mod tests;
