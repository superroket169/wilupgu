use crate::core::dtype::DataKind::F32;
use crate::core::shader::BindingRole::{Accumulate, InOut, Input, Output};
use crate::core::shader::{
    CudaCode, MetaField, MetaType, NativeCode, Shader, ShaderCode, WgslCode,
};

#[path = "../shader-codes/native"]
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
            concat!("src/shader-codes/wgsl/", stringify!($name), ".wgsl"),
            include_str!(concat!("../shader-codes/wgsl/", stringify!($name), ".wgsl")),
        ))
    };
    (cuda, $name:ident) => {
        Some(CudaCode::new(
            concat!("src/shader-codes/cuda/", stringify!($name), ".cu"),
            include_str!(concat!("../shader-codes/cuda/", stringify!($name), ".cu")),
        ))
    };
    (native, $name:ident) => {
        Some(NativeCode::new(native::$name::$name))
    };
}

// A format listed twice is a "field specified more than once" compile error.
macro_rules! builtins {
    ($( $static:ident / $name:ident [$($field:expr),* $(,)?] [$($role:expr),* $(,)?] { $($format:ident)* } )*) => {
        $(
            pub static $static: Shader = Shader {
                name: stringify!($name),
                meta: &[$($field),*],
                layout: &[$($role),*],
                shader_code: ShaderCode {
                    $( $format: code!($format, $name), )*
                    ..ShaderCode::NONE
                },
            };
        )*

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

builtins! {
    // STATIC / name             [meta]                              [tensors]                                      { formats }

    // linear algebra
    MATMUL / matmul               [uint("M"), uint("N"), uint("K")]   [Input(F32), Input(F32), Output(F32)]          { wgsl native }
    MATMUL_ADD / matmul_add       [uint("M"), uint("N"), uint("K")]   [Input(F32), Input(F32), Accumulate(F32)]      { wgsl native }
    MATMUL_TRP / matmul_trp       [uint("M"), uint("N"), uint("K")]   [Input(F32), Input(F32), Output(F32)]          { wgsl native }
    GEMV / gemv                   [uint("M"), uint("N"), uint("K")]   [Input(F32), Input(F32), Output(F32)]          { wgsl native }
    GEMV_ADD / gemv_add           [uint("M"), uint("N"), uint("K")]   [Input(F32), Input(F32), Accumulate(F32)]      { wgsl native }
    TRANSPOSE / transpose         [uint("rows"), uint("cols")]        [Input(F32), Output(F32)]                      { wgsl native }
    DOT / dot                     [uint("n")]                         [Input(F32), Input(F32), Output(F32)]          { wgsl native }

    // elementwise
    ADD / add                     []                                  [Accumulate(F32), Input(F32)]                  { wgsl native cuda }
    MUL / mul                     []                                  [InOut(F32), Input(F32)]                       { wgsl native }
    SCALE / scale                 [float("factor")]                   [InOut(F32)]                                   { wgsl native }
    MAX / max                     []                                  [InOut(F32), Input(F32)]                       { wgsl native }
    MIN / min                     []                                  [InOut(F32), Input(F32)]                       { wgsl native }
    CLAMP / clamp                 [float("lo"), float("hi")]          [InOut(F32)]                                   { wgsl native }

    // reduction
    SUM / sum                     [uint("n")]                         [Input(F32), Output(F32)]                      { wgsl native }

    // buffer init
    ZERO_TENSOR / zero_tensor     [uint("n")]                         [Output(F32)]                                  { wgsl native cuda }
    FILL_CONSTANT / fill_constant [uint("n"), float("value")]         [Output(F32)]                                  { wgsl native }
    FILL_RANDOM / fill_random     [uint("n"), uint("seed")]           [Output(F32)]                                  { wgsl native }
}

#[cfg(test)]
#[path = "../tests/builtins.rs"]
mod tests;
