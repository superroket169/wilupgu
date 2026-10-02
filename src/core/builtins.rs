use crate::core::dtype::DataKind::F32;
use crate::core::shader::BindingRole::{self, Accumulate, InOut, Input, Output};
use crate::core::shader::MetaField::{Float, Uint};
use crate::core::shader::{CudaCode, MetaKind, MetaSlot, NativeCode, Shader, ShaderCode, WgslCode};

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
    ($( $static:ident / $name:ident [$($role:expr),* $(,)?] { $($format:ident)* } )*) => {
        $(
            pub static $static: Shader = Shader {
                name: stringify!($name),
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

const fn meta(fields: MetaSlot) -> BindingRole {
    BindingRole::Meta {
        fields,
        kind: MetaKind::Static,
    }
}

builtins! {
    // linear algebra
    MATMUL / matmul         [Input(F32), Input(F32), Output(F32), meta(&[Uint, Uint, Uint])]     { wgsl native }
    MATMUL_ADD / matmul_add [Input(F32), Input(F32), Accumulate(F32), meta(&[Uint, Uint, Uint])] { wgsl native }
    MATMUL_TRP / matmul_trp [Input(F32), Input(F32), Output(F32), meta(&[Uint, Uint, Uint])]     { wgsl native }
    GEMV / gemv             [Input(F32), Input(F32), Output(F32), meta(&[Uint, Uint, Uint])]     { wgsl native }
    GEMV_ADD / gemv_add     [Input(F32), Input(F32), Accumulate(F32), meta(&[Uint, Uint, Uint])] { wgsl native }
    TRANSPOSE / transpose   [Input(F32), Output(F32), meta(&[Uint, Uint])]                       { wgsl native }
    DOT / dot               [Input(F32), Input(F32), Output(F32), meta(&[Uint])]                 { wgsl native }

    // elementwise
    ADD / add               [Accumulate(F32), Input(F32)]                                        { wgsl native cuda }
    MUL / mul               [InOut(F32), Input(F32)]                                             { wgsl native }
    SCALE / scale           [InOut(F32), meta(&[Float])]                                         { wgsl native }
    MAX / max               [InOut(F32), Input(F32)]                                             { wgsl native }
    MIN / min               [InOut(F32), Input(F32)]                                             { wgsl native }
    CLAMP / clamp           [InOut(F32), meta(&[Float, Float])]                                  { wgsl native }

    // reduction
    SUM / sum               [Input(F32), Output(F32), meta(&[Uint])]                             { wgsl native }

    // buffer init
    ZERO_TENSOR / zero_tensor     [Output(F32), meta(&[Uint])]                                   { wgsl native cuda }
    FILL_CONSTANT / fill_constant [Output(F32), meta(&[Uint, Float])]                            { wgsl native }
    FILL_RANDOM / fill_random     [Output(F32), meta(&[Uint, Uint])]                             { wgsl native }
}

#[cfg(test)]
#[path = "../tests/builtins.rs"]
mod tests;
