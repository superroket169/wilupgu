//! Blueprints: what a tensor or a dispatch will be, before anything exists
//! on a device. Nothing here touches a backend.

use crate::id::GlobalId;
use crate::mesh::Parallelity;
use crate::resolver::Resolvable;
use crate::traits::{BindingRole, DataKind, DataType, HostData, Shader, Workgroups};

pub type TensorId = GlobalId<TensorSpec>;
pub type NodeId = GlobalId<NodeSpec>;

pub enum TensorSize {
    Fixed(u32),
    /// Not known in compile time
    /// whoever resolves it decides HOW based on the owning NodeSpec's `Parallelity`
    Resolvable {
        size: Resolvable<u32>,
        multiplier: u32,
        coefficient: u32,
    },
}

pub enum InitRecipe {
    UploadFromHost(HostData),
    Zero,
}

pub struct TensorSpec {
    id: TensorId,
    kind: DataKind,
    size: TensorSize,
    init: Option<InitRecipe>,
}

impl TensorSpec {
    pub fn blank<D: DataType>(size: TensorSize) -> Self {
        Self::with::<D>(size, None)
    }

    pub fn seeded<D: DataType>(size: TensorSize, data: Vec<D::HostRepr>) -> Self {
        Self::with::<D>(size, Some(InitRecipe::UploadFromHost(D::wrap(data))))
    }

    pub fn zeroed<D: DataType>(size: TensorSize) -> Self {
        Self::with::<D>(size, Some(InitRecipe::Zero))
    }

    fn with<D: DataType>(size: TensorSize, init: Option<InitRecipe>) -> Self {
        Self {
            id: GlobalId::new(),
            kind: D::KIND,
            size,
            init,
        }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn kind(&self) -> DataKind {
        self.kind
    }

    pub fn size(&self) -> &TensorSize {
        &self.size
    }

    pub fn init(&self) -> Option<&InitRecipe> {
        self.init.as_ref()
    }
}

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
    parallelity: Parallelity,
}

impl NodeSpec {
    pub fn new(
        shader: &'static Shader,
        bindings: Vec<Binding>,
        workgroups: Workgroups,
        parallelity: Parallelity,
    ) -> Self {
        Self {
            id: GlobalId::new(),
            shader,
            bindings,
            workgroups,
            parallelity,
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

    pub fn parallelity(&self) -> Parallelity {
        self.parallelity
    }
}
