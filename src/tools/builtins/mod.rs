use crate::backend::dtype::DataKind::F32;
use crate::backend::shader::BindingRole::{Accumulate, InOut, Input, Output};
use crate::backend::shader::{
    CudaCode, MetaField, MetaType, NativeCode, Shader, ShaderCode, WgslCode,
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
    ($( $static:ident / $name:ident [$($field:expr),* $(,)?] [$($role:expr),* $(,)?] $workgroup_size:tt { $($format:ident)* } )*) => {
        $(
            pub static $static: Shader = Shader {
                name: stringify!($name),
                meta: &[$($field),*],
                layout: &[$($role),*],
                workgroup_size: $workgroup_size,
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
    // STATIC / name             [meta]                                    [tensors]                                   workgroup     { formats }

    // linear algebra
    MATMUL / matmul               [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [16, 16, 1]   { wgsl native cuda }
    MATMUL_ADD / matmul_add       [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Accumulate(F32)]   [16, 16, 1]   { wgsl native cuda }
    MATMUL_TRP / matmul_trp       [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [16, 16, 1]   { wgsl native cuda }
    GEMV / gemv                   [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Output(F32)]       [256, 1, 1]   { wgsl native cuda }
    GEMV_ADD / gemv_add           [uint("M"), uint("N"), uint("K")]         [Input(F32), Input(F32), Accumulate(F32)]   [256, 1, 1]   { wgsl native cuda }
    TRANSPOSE / transpose         [uint("rows"), uint("cols")]              [Input(F32), Output(F32)]                   [16, 16, 1]   { wgsl native cuda }
    DOT / dot                     [uint("n")]                               [Input(F32), Input(F32), Output(F32)]       [256, 1, 1]   { wgsl native cuda }

    // elementwise
    ADD / add                     [uint("n")]                               [Accumulate(F32), Input(F32)]               [256, 1, 1]   { wgsl native cuda }
    MUL / mul                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   { wgsl native cuda }
    SCALE / scale                 [uint("n"), float("factor")]              [InOut(F32)]                                [256, 1, 1]   { wgsl native cuda }
    MAX / max                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   { wgsl native cuda }
    MIN / min                     [uint("n")]                               [InOut(F32), Input(F32)]                    [256, 1, 1]   { wgsl native cuda }
    CLAMP / clamp                 [uint("n"), float("lo"), float("hi")]     [InOut(F32)]                                [256, 1, 1]   { wgsl native cuda }

    // reduction
    SUM / sum                     [uint("n")]                               [Input(F32), Output(F32)]                   [256, 1, 1]   { wgsl native cuda }

    // buffer init
    ZERO_TENSOR / zero_tensor     [uint("n")]                               [Output(F32)]                               [256, 1, 1]   { wgsl native cuda }
    FILL_CONSTANT / fill_constant [uint("n"), float("value")]               [Output(F32)]                               [256, 1, 1]   { wgsl native cuda }
    FILL_RANDOM / fill_random     [uint("n"), uint("seed")]                 [Output(F32)]                               [256, 1, 1]   { wgsl native cuda }
}

#[cfg(test)]
#[path = "../../tests/builtins.rs"]
mod tests;
