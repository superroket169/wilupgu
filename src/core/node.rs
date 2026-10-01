//! One dispatch's blueprint: what shader, wired to which tensors, at what
//! size. Nothing here touches a backend.

use crate::core::id::GlobalId;
use crate::core::shader::{BindingRole, Shader, Workgroups};
use crate::core::tensor::TensorId;

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

/// Fields are private and set only through `new`
/// a spec doesn't change after it's made.
pub struct NodeSpec {
    id: NodeId,
    shader: &'static Shader,
    bindings: Vec<Binding>,
    workgroups: Workgroups,
}

impl NodeSpec {
    pub fn new(shader: &'static Shader, bindings: Vec<Binding>, workgroups: Workgroups) -> Self {
        Self {
            id: GlobalId::new(),
            shader,
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

    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    pub fn workgroups(&self) -> Workgroups {
        self.workgroups
    }
}
