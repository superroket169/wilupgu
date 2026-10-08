use crate::backend::dtype::DataKind;

/// What a shader does with one tensor slot, and the kind the slot holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingRole {
    /// Only read.
    Input(DataKind),
    /// Written; its old contents don't matter.
    Output(DataKind),
    /// Read, then written in place.
    InOut(DataKind),
    /// Added to: the old contents stay and the shader adds its result.
    Accumulate(DataKind),
}

/// One shader: its name, meta, tensor layout, workgroup size and code in each format.
pub struct Shader {
    /// The shader's name; builtins name their code files after it.
    pub name: &'static str,
    /// Meta is always at slot 0
    pub meta: &'static [MetaField],
    /// tensors are always at slots 1..=layout.len()
    pub layout: &'static [BindingRole],
    /// Threads per workgroup. WGSL's `@workgroup_size` must match; CUDA
    /// launches with it as `blockDim`.
    pub workgroup_size: [u32; 3],
    /// The shader's code, in each format it has.
    pub shader_code: ShaderCode,
}

/// The type of one meta field. Both are one 32-bit word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaType {
    /// `u32`.
    Uint,
    /// `f32`, passed as its bits in a `u32` word.
    Float,
}

/// One field of a shader's meta: its name and type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MetaField {
    /// The field's name, as in the shader source.
    pub name: &'static str,
    /// The field's type.
    pub ty: MetaType,
}

/// How many workgroups one dispatch runs, in each dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroups {
    /// Workgroups along x.
    pub x: u32,
    /// Workgroups along y.
    pub y: u32,
    /// Workgroups along z.
    pub z: u32,
}

impl Workgroups {
    /// `n` workgroups along x, 1 along y and z.
    pub const fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    pub(crate) fn dims(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

/// One tensor slot's bytes, given to native code: the CPU's counterpart of
/// [`Access`](crate::backend::Access).
pub enum NativeBinding<'a> {
    /// The slot is only read.
    Read(&'a [u8]),
    /// The slot is written, and may be read too.
    Write(&'a mut [u8]),
}

/// A shader's code, keyed by code format rather than by backend
///
/// wgpu and vulkano both take `Wgsl`
/// cpu and rayon both take `Native`.
pub struct ShaderCode {
    /// WGSL code.
    pub wgsl: Option<WgslCode>,
    /// Native Rust code.
    pub native: Option<NativeCode>,
    /// CUDA C++ code.
    pub cuda: Option<CudaCode>,
}

impl ShaderCode {
    /// No code in any format; a base for `..ShaderCode::NONE`.
    pub const NONE: Self = Self {
        wgsl: None,
        native: None,
        cuda: None,
    };

    /// True if there is code in `format`.
    pub fn has(&self, format: ShaderFormat) -> bool {
        match format {
            ShaderFormat::Wgsl => self.wgsl.is_some(),
            ShaderFormat::Native => self.native.is_some(),
            ShaderFormat::Cuda => self.cuda.is_some(),
        }
    }
}

/// A shader code format. Each backend runs one, its `Dispatch::FORMAT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderFormat {
    /// WGSL source.
    Wgsl,
    /// A Rust function.
    Native,
    /// CUDA C++ source.
    Cuda,
}

/// WGSL source and the path of the file it came from.
#[derive(Debug, Clone, Copy)]
pub struct WgslCode {
    path: &'static str,
    source: &'static str,
}

impl WgslCode {
    /// Panics if `path` doesn't end in `.wgsl`; in a `const` or `static`,
    /// that is a compile error.
    pub const fn new(path: &'static str, source: &'static str) -> Self {
        assert!(
            ends_with(path, ".wgsl"),
            "WGSL code must come from a .wgsl file"
        );
        Self { path, source }
    }

    /// The file the source came from.
    pub fn path(&self) -> &'static str {
        self.path
    }

    /// The source text.
    pub fn source(&self) -> &'static str {
        self.source
    }
}

/// CUDA C++ source and the path of the file it came from.
#[derive(Debug, Clone, Copy)]
pub struct CudaCode {
    path: &'static str,
    source: &'static str,
}

impl CudaCode {
    /// Panics if `path` doesn't end in `.cu`; in a `const` or `static`, that
    /// is a compile error.
    pub const fn new(path: &'static str, source: &'static str) -> Self {
        assert!(
            ends_with(path, ".cu"),
            "CUDA code must come from a .cu file"
        );
        Self { path, source }
    }

    /// The file the source came from.
    pub fn path(&self) -> &'static str {
        self.path
    }

    /// The source text.
    pub fn source(&self) -> &'static str {
        self.source
    }
}

/// Native code: a Rust function that takes the meta words and one
/// `NativeBinding` per tensor slot, in slot order (index 0 is slot 1).
#[derive(Clone, Copy)]
pub struct NativeCode(fn(&[u32], &mut [NativeBinding]));

impl NativeCode {
    /// Wraps `f`.
    pub const fn new(f: fn(&[u32], &mut [NativeBinding])) -> Self {
        Self(f)
    }

    /// Runs the code on `meta` and `bindings`.
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
