//! One dispatch's blueprint: what shader, wired to which tensors, at what
//! size. Nothing here touches a backend.

use crate::backend::shader::{BindingRole, MetaType, Shader, Workgroups};
use crate::tools::core::deferred::{Dynamic, Resolvable};
use crate::tools::core::id::GlobalId;
use crate::tools::core::id::TensorId;

pub type NodeId = GlobalId<NodeSpec>;

#[derive(Debug, Clone)]
pub struct Binding {
    pub slot: u32,
    pub tensor: TensorId,
    pub mode: BindingRole,
}

impl Binding {
    #[inline]
    pub fn new(slot: u32, tensor: TensorId, mode: BindingRole) -> Self {
        Self { slot, tensor, mode }
    }
}

pub enum MetaSource<T> {
    /// Never changes. build time constant
    Once(Resolvable<T>),
    /// could be changable by user.
    /// example: adamw shudeler step.
    PerRun(Dynamic<T>),
}

impl<T> Clone for MetaSource<T> {
    fn clone(&self) -> Self {
        match self {
            MetaSource::Once(r) => MetaSource::Once(r.clone()),
            MetaSource::PerRun(d) => MetaSource::PerRun(d.clone()),
        }
    }
}

/// One meta field's value; the shader's `MetaField` gives its name and type.
#[derive(Clone)]
pub enum MetaValue {
    Uint(MetaSource<u32>),
    Float(MetaSource<f32>),
}

impl MetaValue {
    pub fn ty(&self) -> MetaType {
        match self {
            MetaValue::Uint(_) => MetaType::Uint,
            MetaValue::Float(_) => MetaType::Float,
        }
    }

    pub fn is_per_run(&self) -> bool {
        matches!(
            self,
            MetaValue::Uint(MetaSource::PerRun(_)) | MetaValue::Float(MetaSource::PerRun(_))
        )
    }

    /// `None` for a per-run value.
    pub(crate) fn once_word(&self) -> Option<u32> {
        match self {
            MetaValue::Uint(MetaSource::Once(r)) => Some(*r.value()),
            MetaValue::Float(MetaSource::Once(r)) => Some(r.value().to_bits()),
            _ => None,
        }
    }

    // Which `Dynamic` this reads, so a run reads each one once however many
    // nodes share it. `None` for a once value.
    pub(crate) fn dynamic_key(&self) -> Option<usize> {
        match self {
            MetaValue::Uint(MetaSource::PerRun(d)) => Some(d.key()),
            MetaValue::Float(MetaSource::PerRun(d)) => Some(d.key()),
            _ => None,
        }
    }

    // Consumes this run's read of a per-run value.
    pub(crate) fn read_word_for_run(&self) -> u32 {
        match self {
            MetaValue::Uint(MetaSource::PerRun(d)) => d.read_for_run(),
            MetaValue::Float(MetaSource::PerRun(d)) => d.read_for_run().to_bits(),
            once => once.once_word().expect("a once value always has a word"),
        }
    }
}

/// Fields are private and set only through `new`
/// a spec doesn't change after it's made.
pub struct NodeSpec {
    id: NodeId,
    shader: &'static Shader,
    meta: Vec<MetaValue>,
    bindings: Vec<Binding>,
    workgroups: Workgroups,
}

impl NodeSpec {
    pub fn new(
        shader: &'static Shader,
        meta: Vec<MetaValue>,
        bindings: Vec<Binding>,
        workgroups: Workgroups,
    ) -> Self {
        Self {
            id: GlobalId::new(),
            shader,
            meta,
            bindings,
            workgroups,
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn shader(&self) -> &'static Shader {
        self.shader
    }

    pub fn meta(&self) -> &[MetaValue] {
        &self.meta
    }

    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    pub fn workgroups(&self) -> Workgroups {
        self.workgroups
    }
}
