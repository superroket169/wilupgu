//! General-purpose GPU math shaders, ready to run.
//! Each one has WGSL, native and CUDA code.
//! The catalog, with every shader's formula, meta, tensor layout and `min_lens`, is
//! [`docs/SHADERS.md`](https://github.com/superroket169/wilupgu/blob/main/docs/SHADERS.md).

use crate::backend::dtype::DataKind::F32;
use crate::backend::shader::BindingRole::{Accumulate, InOut, Input, Output};

use crate::backend::shader::{
    Cuda, MetaField, MetaType, Native, Shader, ShaderBinary, Signature, Wgsl, Workgroups,
};
use crate::backend::Dispatch;
use crate::error::Error;

#[path = "codes/native"]
mod native {
    mod common;

    pub(super) mod add;
    pub(super) mod clamp;
    pub(super) mod dot;
    pub(super) mod fill_constant;
    pub(super) mod fill_random;
    pub(super) mod gemv;
    pub(super) mod gemv_add;
    pub(super) mod matmul;
    pub(super) mod matmul_add;
    pub(super) mod matmul_trp;
    pub(super) mod max;
    pub(super) mod min;
    pub(super) mod mul;
    pub(super) mod scale;
    pub(super) mod sum;
    pub(super) mod transpose;
    pub(super) mod zero_tensor;
}

/// A shader with code in every format: how it is called, then WGSL, CUDA and
/// native code. [`PortableShader::shader_for`] picks the format a backend
/// runs. The builtins are made with [`shader_table!`](crate::shader_table), which other crates can
/// use for their own shaders.
pub struct PortableShader {
    /// How it is called.
    pub signature: Signature,
    /// WGSL source.
    pub wgsl: Wgsl,
    /// CUDA C++ source.
    pub cuda: Cuda,
    /// Native code.
    pub native: Native,
}

impl PortableShader {
    /// The [`Shader`] for backend `B`: native or WGSL code as it is, or the
    /// CUDA code compiled to PTX with NVRTC when `compile-nvrtc` is on. Fails
    /// with [`Error::UnsupportedBinary`] if `B` runs none of these.
    pub fn shader_for<B: Dispatch>(&'static self) -> Result<Shader, Error> {
        let ready = [
            ShaderBinary::Native(self.native),
            ShaderBinary::Wgsl(self.wgsl.clone()),
        ];
        if let Some(binary) = ready.into_iter().find(B::accepts) {
            return Ok(Shader::new(&self.signature, binary));
        }
        #[cfg(feature = "compile-nvrtc")]
        {
            use crate::backend::shader::Ptx;
            use crate::tools::compile::{compilers::nvrtc::Nvrtc, ShaderCompiler};
            use std::borrow::Cow;
            if B::accepts(&ShaderBinary::Ptx(Ptx(Cow::Borrowed("")))) {
                let ptx = Nvrtc::default().compile(&self.cuda)?;
                return Ok(Shader::new(&self.signature, ShaderBinary::Ptx(ptx)));
            }
        }
        Err(Error::UnsupportedBinary {
            shader: self.signature.name,
        })
    }
}

/// Defines a table of [`PortableShader`]s: a `pub static` for each row, and
/// `pub static ALL` listing them in row order.
///
/// A row is `STATIC / name [meta] [layout] [x, y, z] min_lens`: the static's
/// name, the shader's name, its [`MetaField`]s, its
/// [`BindingRole`](crate::backend::shader::BindingRole)s, its workgroup size
/// and its `min_lens` function, by name.
///
/// The code is found from `name`, so file names, the native function and the
/// shader's name can't drift apart. Paths are relative to the file that calls
/// the macro:
/// - `codes/wgsl/<name>.wgsl`
/// - `codes/cuda/<name>.cu`
/// - the native function `native::<name>::entry`, so a `native` module with
///   one submodule per shader must be in scope there.
///
/// ```ignore
/// use wilupgu::backend::dtype::DataKind::F32;
/// use wilupgu::backend::shader::{BindingRole::*, MetaField, MetaType, Workgroups};
///
/// const N: MetaField = MetaField { name: "n", ty: MetaType::Uint };
///
/// fn each_n(m: &[u32], _: Workgroups, l: &mut [usize]) {
///     l.fill(m[0] as usize);
/// }
///
/// mod native {
///     pub mod axpy; // has `pub fn entry(meta: &[u32], bindings: &mut [NativeBinding])`
/// }
///
/// wilupgu::shader_table! {
///     AXPY / axpy [N] [Accumulate(F32), Input(F32)] [256, 1, 1] each_n
/// }
/// ```
#[macro_export]
macro_rules! shader_table {
    ($( $static:ident / $name:ident [$($field:expr),* $(,)?] [$($role:expr),* $(,)?] $workgroup_size:tt $min_lens:ident )*) => {
        $(
            #[doc = concat!("The `", stringify!($name), "` shader.")]
            pub static $static: $crate::tools::builtins::PortableShader = $crate::tools::builtins::PortableShader {
                signature: $crate::backend::shader::Signature {
                    name: stringify!($name),
                    meta: &[$($field),*],
                    layout: &[$($role),*],
                    min_lens: $min_lens,
                    workgroup_size: $workgroup_size,
                },
                wgsl: $crate::backend::shader::Wgsl(::std::borrow::Cow::Borrowed(include_str!(concat!("codes/wgsl/", stringify!($name), ".wgsl")))),
                cuda: $crate::backend::shader::Cuda(::std::borrow::Cow::Borrowed(include_str!(concat!("codes/cuda/", stringify!($name), ".cu")))),
                native: $crate::backend::shader::Native(native::$name::entry),
            };
        )*

        /// Every shader of this table, in row order.
        pub static ALL: &[&$crate::tools::builtins::PortableShader] = &[$(&$static),*];
    };
}

const fn uint(name: &'static str) -> MetaField {
    MetaField {
        name,
        ty: MetaType::Uint,
    }
}

const fn float(name: &'static str) -> MetaField {
    MetaField {
        name,
        ty: MetaType::Float,
    }
}

// `min_lens` of the builtins, in elements; meta words are in table order.
// Products saturate: a length past `usize::MAX` is one no buffer holds, so
// the safe layer's check fails instead of passing a short buffer after a wrap.

// Every slot holds `n` elements.
fn each_n(m: &[u32], _: Workgroups, l: &mut [usize]) {
    l.fill(m[0] as usize);
}

// A: MxK, B: KxN (or NxK for matmul_trp), C: MxN.
fn matmul_lens(m: &[u32], _: Workgroups, l: &mut [usize]) {
    let (rows, cols, inner) = (m[0] as usize, m[1] as usize, m[2] as usize);
    l.copy_from_slice(&[
        rows.saturating_mul(inner),
        inner.saturating_mul(cols),
        rows.saturating_mul(cols),
    ]);
}

// One row: A: 1xK, B: KxN, C: 1xN; M is ignored.
fn gemv_lens(m: &[u32], _: Workgroups, l: &mut [usize]) {
    let (cols, inner) = (m[1] as usize, m[2] as usize);
    l.copy_from_slice(&[inner, inner.saturating_mul(cols), cols]);
}

fn transpose_lens(m: &[u32], _: Workgroups, l: &mut [usize]) {
    l.fill((m[0] as usize).saturating_mul(m[1] as usize));
}

// Inputs hold `n`; the last slot holds one partial per workgroup. Native code
// runs as one workgroup and always writes `partial[0]`, hence at least 1.
fn reduce_lens(m: &[u32], wg: Workgroups, l: &mut [usize]) {
    let partials = (wg.x as usize)
        .saturating_mul(wg.y as usize)
        .saturating_mul(wg.z as usize)
        .max(1);
    l.fill(m[0] as usize);
    *l.last_mut().expect("a reduction has an output slot") = partials;
}

shader_table! {
    // STATIC / name             [meta]                                    [tensors]                                   workgroup     min_lens

    // linear algebra
    MATMUL / matmul               [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [16, 16, 1]   matmul_lens
    MATMUL_ADD / matmul_add       [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Accumulate(F32)]   [16, 16, 1]   matmul_lens
    MATMUL_TRP / matmul_trp       [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [16, 16, 1]   matmul_lens
    GEMV / gemv                   [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [256, 1, 1]   gemv_lens
    GEMV_ADD / gemv_add           [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Accumulate(F32)]   [256, 1, 1]   gemv_lens
    TRANSPOSE / transpose         [uint("rows"), uint("cols")]              [Input(F32), Output(F32)]                   [16, 16, 1]   transpose_lens
    DOT / dot                     [uint("n")]                               [Input(F32), Input(F32), Output(F32)]       [256, 1, 1]   reduce_lens

    // elementwise
    ADD / add                     [uint("n")]                               [Accumulate(F32), Input(F32)]               [256, 1, 1]   each_n
    MUL / mul                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   each_n
    SCALE / scale                 [uint("n"), float("factor")]              [InOut(F32)]                                [256, 1, 1]   each_n
    MAX / max                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   each_n
    MIN / min                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   each_n
    CLAMP / clamp                 [uint("n"), float("lo"), float("hi")]     [InOut(F32)]                                [256, 1, 1]   each_n

    // reduction
    SUM / sum                     [uint("n")]                               [Input(F32), Output(F32)]                   [256, 1, 1]   reduce_lens

    // buffer init
    ZERO_TENSOR / zero_tensor     [uint("n")]                               [Output(F32)]                               [256, 1, 1]   each_n
    FILL_CONSTANT / fill_constant [uint("n"), float("value")]               [Output(F32)]                               [256, 1, 1]   each_n
    FILL_RANDOM / fill_random     [uint("n"), uint("seed")]                 [Output(F32)]                               [256, 1, 1]   each_n
}

#[cfg(test)]
#[path = "../../tests/builtins.rs"]
mod tests;
