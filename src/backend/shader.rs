use std::borrow::Cow;

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

impl BindingRole {
    /// The kind the slot holds.
    pub fn kind(&self) -> DataKind {
        let (BindingRole::Input(kind)
        | BindingRole::Output(kind)
        | BindingRole::InOut(kind)
        | BindingRole::Accumulate(kind)) = self;
        *kind
    }
}

/// How a shader is called: its name, meta, tensor layout, `min_lens` and
/// workgroup size. The same for every binary of the shader.
#[derive(Debug)]
pub struct Signature {
    /// The shader's name; builtins name their code files after it.
    pub name: &'static str,
    /// The meta's fields, in order; the meta is always slot 0.
    pub meta: &'static [MetaField],
    /// One role per tensor slot; tensors are slots `1..=layout.len()`.
    pub layout: &'static [BindingRole],
    /// The elements each tensor slot needs, from the meta words and the workgroup count:
    /// writes one entry per layout slot into `lens`.
    /// No code of the shader reads or writes a slot past it.
    pub min_lens: fn(meta: &[u32], workgroups: Workgroups, lens: &mut [usize]),
    /// Threads per workgroup. WGSL's `@workgroup_size` must match; CUDA
    /// launches with it as `blockDim`.
    pub workgroup_size: [u32; 3],
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
    pub ty: MetaType,
}

/// How many workgroups one dispatch runs, in each dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroups {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Workgroups {
    /// `n` workgroups along x, 1 along y and z.
    pub const fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    /// Enough workgroups along x for `len` threads of `shader`, one per
    /// element: `len` over the shader's workgroup width, rounded up. 0 for a
    /// `len` of 0, which a node rejects.
    pub fn cover(len: u32, shader: &Shader) -> Self {
        Self::linear(len.div_ceil(shader.signature().workgroup_size[0]))
    }

    pub(crate) fn dims(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

/// One tensor slot's bytes, given to native code: the CPU's counterpart of
/// [`Access`](crate::backend::Access).
pub enum NativeBinding<'a> {
    Read(&'a [u8]),
    /// The slot is written, and may be read too.
    Write(&'a mut [u8]),
}

/// WGSL source.
#[derive(Debug, Clone)]
pub struct Wgsl(pub Cow<'static, str>);

#[derive(Debug, Clone)]
pub struct Glsl(pub Cow<'static, str>);

#[derive(Debug, Clone)]
pub struct Spirv(pub Cow<'static, [u32]>);

/// CUDA C++ source. No backend takes it; it is compiled to [`Ptx`] or [`Cubin`].
#[derive(Debug, Clone)]
pub struct Cuda(pub Cow<'static, str>);

/// PTX, which the CUDA driver compiles for the device it loads it on.
#[derive(Debug, Clone)]
pub struct Ptx(pub Cow<'static, str>);

/// A cubin: machine code for one CUDA architecture.
#[derive(Debug, Clone)]
pub struct Cubin(pub Cow<'static, [u8]>);

/// Native code: a Rust function that takes the meta words and one
/// `NativeBinding` per tensor slot, in slot order (index 0 is slot 1).
#[derive(Debug, Clone, Copy)]
pub struct Native(pub fn(&[u32], &mut [NativeBinding]));

/// A shader's code in a form a backend takes. Each backend says which ones
/// with `Dispatch::accepts`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ShaderBinary {
    Wgsl(Wgsl),
    Glsl(Glsl),
    Spirv(Spirv),
    Ptx(Ptx),
    Cubin(Cubin),
    Native(Native),
}

/// A shader to build nodes from: its [`Signature`] and one binary.
#[derive(Debug, Clone)]
pub struct Shader {
    signature: &'static Signature,
    binary: ShaderBinary,
}

impl Shader {
    /// `binary` must be the code `signature` describes: the safe layer checks
    /// every run against the signature, and the code must keep to it (its
    /// slots, its meta, its `min_lens`). Nothing checks the pair itself.
    pub fn new(signature: &'static Signature, binary: ShaderBinary) -> Self {
        Self { signature, binary }
    }

    /// How it is called.
    pub fn signature(&self) -> &'static Signature {
        self.signature
    }

    /// Its code.
    pub fn binary(&self) -> &ShaderBinary {
        &self.binary
    }
}
