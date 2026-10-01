use crate::core::dtype::DataKind;
use crate::core::resolver::Resolvable;

#[derive(Debug, Clone)]
pub enum BindingRole {
    Input(DataKind),
    Output(DataKind),
    InOut(DataKind),
    Accumulate(DataKind),
    Meta { fields: MetaSlot, kind: MetaKind },
}

impl BindingRole {
    pub(crate) fn accepts(&self, actual: &BindingRole) -> bool {
        match (self, actual) {
            (BindingRole::Input(a), BindingRole::Input(b)) => a == b,
            (BindingRole::Output(a), BindingRole::Output(b)) => a == b,
            (BindingRole::InOut(a), BindingRole::InOut(b)) => a == b,
            (BindingRole::Accumulate(a), BindingRole::Accumulate(b)) => a == b,
            (BindingRole::Meta { .. }, BindingRole::Meta { .. }) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetaKind {
    Static,
    Dynamic,
    Maximized(ResolvedSize),
}

/// A size not known at compile time
/// shared by `TensorSize` and `MetaKind`
/// so a tensor and its meta can be given the same `Resolvable`
///
/// resolving one then resolves both, ruling out the two ever disagreeing on it.
#[derive(Debug, Clone)]
pub struct ResolvedSize {
    pub size: Resolvable<u32>,
    pub multiplier: u32,
    pub coefficient: u32,
}

pub struct Shader {
    pub name: &'static str,
    pub layout: &'static [BindingRole],
    pub shader_code: &'static [ShaderCode],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaField {
    Uint,
    Float,
}

pub type MetaSlot = &'static [MetaField];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroups {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Workgroups {
    pub const fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    pub(crate) fn dims(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

pub type CpuBuffer = std::sync::Arc<std::sync::Mutex<Vec<u8>>>;

#[derive(Clone)]
pub struct CpuBinding {
    pub slot: u32,
    pub buffer: CpuBuffer,
}

/// A shader's code, keyed by code format rather than by backend
///
/// wgpu and vulkano both take `Wgsl`
/// cpu and rayon both take `Native`.
pub enum ShaderCode {
    Wgsl(&'static str),
    #[cfg(any(feature = "cpu", feature = "rayon"))]
    Native(fn(&[CpuBinding])),
    // TODO: Cuda(...) - needs a real dispatch type once the CUDA backend is written
}
