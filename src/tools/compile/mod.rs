//! The compile tool: [`ShaderCompiler`], turning a shader's code from one
//! form into another, like CUDA C into PTX. Each compiler is its own type,
//! behind its own feature, and implements it once for each pair of forms it
//! handles.
//!
//! It knows nothing of a shader's [`Signature`](crate::backend::shader::Signature);
//! pairing the compiled code with it is the caller's
//! ([`Shader::new`](crate::backend::shader::Shader::new)).

use crate::error::Error;

/// The compilers, each behind its own feature.
pub mod compilers {
    #[cfg(feature = "compile-nvrtc")]
    pub mod nvrtc;
}

/// Turns code in the form `In` into the form `Out`.
pub trait ShaderCompiler<In, Out> {
    /// Compiles `input`. Fails with the compiler's own report.
    fn compile(&self, input: &In) -> Result<Out, Error>;
}
