use crate::backend::dtype::DataKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingRole {
    Input(DataKind),
    Output(DataKind),
    InOut(DataKind),
    Accumulate(DataKind),
}

pub struct Shader {
    pub name: &'static str,
    /// Meta is always at slot 0
    pub meta: &'static [MetaField],
    /// tensors are always at slots 1..=layout.len()
    pub layout: &'static [BindingRole],
    // WGSL's `@workgroup_size` must match; CUDA launches with it as `blockDim`.
    pub workgroup_size: [u32; 3],
    pub shader_code: ShaderCode,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaType {
    Uint,
    Float,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MetaField {
    pub name: &'static str,
    pub ty: MetaType,
}

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

pub enum NativeBinding<'a> {
    Read(&'a [u8]),
    Write(&'a mut [u8]),
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
pub struct NativeCode(fn(&[u32], &mut [NativeBinding]));

impl NativeCode {
    pub const fn new(f: fn(&[u32], &mut [NativeBinding])) -> Self {
        Self(f)
    }

    pub fn run(&self, meta: &[u32], bindings: &mut [NativeBinding]) {
        (self.0)(meta, bindings)
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
