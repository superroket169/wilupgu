//! General-purpose GPU math shaders, ready to run.
//! Each one has WGSL, native and CUDA code.
//! The catalog, with every shader's formula, meta, tensor layout and `min_lens`, is
//! [`docs/SHADERS.md`](https://github.com/superroket169/wilupgu/blob/main/docs/SHADERS.md).

use crate::backend::dtype::DataKind::F32;
use crate::backend::shader::BindingRole::{Accumulate, InOut, Input, Output};
use crate::backend::shader::{
    CudaCode, MetaField, MetaType, NativeCode, Shader, ShaderCode, WgslCode, Workgroups,
};

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

// One format's code for the shader `$name`. File path, file name and native fn
// name all come from `$name`, so they can't drift apart.
macro_rules! code {
    (wgsl, $name:ident) => {
        Some(WgslCode::new(
            concat!("src/tools/builtins/codes/wgsl/", stringify!($name), ".wgsl"),
            include_str!(concat!("codes/wgsl/", stringify!($name), ".wgsl")),
        ))
    };
    (cuda, $name:ident) => {
        Some(CudaCode::new(
            concat!("src/tools/builtins/codes/cuda/", stringify!($name), ".cu"),
            include_str!(concat!("codes/cuda/", stringify!($name), ".cu")),
        ))
    };
    (native, $name:ident) => {
        Some(NativeCode::new(native::$name::entry))
    };
}

// A format listed twice is a "field specified more than once" compile error.
macro_rules! builtins {
    ($( $static:ident / $name:ident [$($field:expr),* $(,)?] [$($role:expr),* $(,)?] $workgroup_size:tt $min_lens:ident { $($format:ident)* } )*) => {
        $(
            #[doc = concat!("The `", stringify!($name), "` builtin; see the catalog in the module docs.")]
            // `..ShaderCode::NONE` fills the formats a shader doesn't list; it is a no-op
            // only while every builtin has all of them.
            #[allow(clippy::needless_update)]
            pub static $static: Shader = Shader {
                name: stringify!($name),
                meta: &[$($field),*],
                layout: &[$($role),*],
                min_lens: $min_lens,
                workgroup_size: $workgroup_size,
                shader_code: ShaderCode {
                    $( $format: code!($format, $name), )*
                    ..ShaderCode::NONE
                },
            };
        )*

        /// Every builtin, in catalog order.
        pub static ALL: &[&Shader] = &[$(&$static),*];
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

// Every slot holds `n` elements.
fn each_n(m: &[u32], _: Workgroups, l: &mut [usize]) {
    l.fill(m[0] as usize);
}

// A: MxK, B: KxN (or NxK for matmul_trp), C: MxN.
fn matmul_lens(m: &[u32], _: Workgroups, l: &mut [usize]) {
    let (rows, cols, inner) = (m[0] as usize, m[1] as usize, m[2] as usize);
    l.copy_from_slice(&[rows * inner, inner * cols, rows * cols]);
}

// One row: A: 1xK, B: KxN, C: 1xN; M is ignored.
fn gemv_lens(m: &[u32], _: Workgroups, l: &mut [usize]) {
    let (cols, inner) = (m[1] as usize, m[2] as usize);
    l.copy_from_slice(&[inner, inner * cols, cols]);
}

fn transpose_lens(m: &[u32], _: Workgroups, l: &mut [usize]) {
    l.fill(m[0] as usize * m[1] as usize);
}

// Inputs hold `n`; the last slot holds one partial per workgroup. Native code
// runs as one workgroup and always writes `partial[0]`, hence at least 1.
fn reduce_lens(m: &[u32], wg: Workgroups, l: &mut [usize]) {
    let partials = (wg.x as usize * wg.y as usize * wg.z as usize).max(1);
    l.fill(m[0] as usize);
    *l.last_mut().expect("a reduction has an output slot") = partials;
}

builtins! {
    // STATIC / name             [meta]                                    [tensors]                                   workgroup     min_lens       { formats }

    // linear algebra
    MATMUL / matmul               [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [16, 16, 1]   matmul_lens    { wgsl native cuda }
    MATMUL_ADD / matmul_add       [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Accumulate(F32)]   [16, 16, 1]   matmul_lens    { wgsl native cuda }
    MATMUL_TRP / matmul_trp       [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [16, 16, 1]   matmul_lens    { wgsl native cuda }
    GEMV / gemv                   [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [256, 1, 1]   gemv_lens      { wgsl native cuda }
    GEMV_ADD / gemv_add           [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Accumulate(F32)]   [256, 1, 1]   gemv_lens      { wgsl native cuda }
    TRANSPOSE / transpose         [uint("rows"), uint("cols")]              [Input(F32), Output(F32)]                   [16, 16, 1]   transpose_lens { wgsl native cuda }
    DOT / dot                     [uint("n")]                               [Input(F32), Input(F32), Output(F32)]       [256, 1, 1]   reduce_lens    { wgsl native cuda }

    // elementwise
    ADD / add                     [uint("n")]                               [Accumulate(F32), Input(F32)]               [256, 1, 1]   each_n         { wgsl native cuda }
    MUL / mul                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   each_n         { wgsl native cuda }
    SCALE / scale                 [uint("n"), float("factor")]              [InOut(F32)]                                [256, 1, 1]   each_n         { wgsl native cuda }
    MAX / max                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   each_n         { wgsl native cuda }
    MIN / min                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   each_n         { wgsl native cuda }
    CLAMP / clamp                 [uint("n"), float("lo"), float("hi")]     [InOut(F32)]                                [256, 1, 1]   each_n         { wgsl native cuda }

    // reduction
    SUM / sum                     [uint("n")]                               [Input(F32), Output(F32)]                   [256, 1, 1]   reduce_lens    { wgsl native cuda }

    // buffer init
    ZERO_TENSOR / zero_tensor     [uint("n")]                               [Output(F32)]                               [256, 1, 1]   each_n         { wgsl native cuda }
    FILL_CONSTANT / fill_constant [uint("n"), float("value")]               [Output(F32)]                               [256, 1, 1]   each_n         { wgsl native cuda }
    FILL_RANDOM / fill_random     [uint("n"), uint("seed")]                 [Output(F32)]                               [256, 1, 1]   each_n         { wgsl native cuda }
}

#[cfg(test)]
#[path = "../../tests/builtins.rs"]
mod tests;
