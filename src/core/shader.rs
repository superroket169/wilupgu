use crate::core::dtype::DataKind;

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
    /// Never changes. build time constant
    Static,
    /// could be changable by user.
    /// example: adamw shudeler step.
    Dynamic,
}

pub struct Shader {
    pub name: &'static str,
    pub layout: &'static [BindingRole],
    pub shader_code: ShaderCode,
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
pub struct ShaderCode {
    pub wgsl: Option<WgslCode>,
    pub native: Option<NativeCode>,
    pub cuda: Option<CudaCode>,
}

impl ShaderCode {
    pub const NONE: Self = Self {
        wgsl: None,
        native: None,
        cuda: None,
    };

    pub fn has(&self, format: ShaderFormat) -> bool {
        match format {
            ShaderFormat::Wgsl => self.wgsl.is_some(),
            ShaderFormat::Native => self.native.is_some(),
            ShaderFormat::Cuda => self.cuda.is_some(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderFormat {
    Wgsl,
    Native,
    Cuda,
}

#[derive(Debug, Clone, Copy)]
pub struct WgslCode {
    path: &'static str,
    source: &'static str,
}

impl WgslCode {
    pub const fn new(path: &'static str, source: &'static str) -> Self {
        assert!(
            ends_with(path, ".wgsl"),
            "WGSL code must come from a .wgsl file"
        );
        Self { path, source }
    }

    pub fn path(&self) -> &'static str {
        self.path
    }

    pub fn source(&self) -> &'static str {
        self.source
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaCode {
    path: &'static str,
    source: &'static str,
}

impl CudaCode {
    pub const fn new(path: &'static str, source: &'static str) -> Self {
        assert!(
            ends_with(path, ".cu"),
            "CUDA code must come from a .cu file"
        );
        Self { path, source }
    }

    pub fn path(&self) -> &'static str {
        self.path
    }

    pub fn source(&self) -> &'static str {
        self.source
    }
}

#[derive(Clone, Copy)]
pub struct NativeCode(fn(&[CpuBinding]));

impl NativeCode {
    pub const fn new(f: fn(&[CpuBinding])) -> Self {
        Self(f)
    }

    pub fn run(&self, bindings: &[CpuBinding]) {
        (self.0)(bindings)
    }
}

// `str::ends_with` isn't callable in a const fn.
const fn ends_with(s: &str, suffix: &str) -> bool {
    let (s, suffix) = (s.as_bytes(), suffix.as_bytes());
    if suffix.len() > s.len() {
        return false;
    }
    let offset = s.len() - suffix.len();
    let mut i = 0;

    while i < suffix.len() {
        if s[offset + i] != suffix[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
#[path = "../tests/shader.rs"]
mod tests;
