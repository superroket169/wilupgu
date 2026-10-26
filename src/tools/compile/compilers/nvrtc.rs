//! [`Nvrtc`]: NVIDIA's runtime compiler, [`Cuda`] into [`Ptx`] or [`Cubin`].
//! The NVRTC library is loaded the first time it compiles; without it,
//! compiling fails. The CUDA version its bindings follow is picked like for
//! any cudarc user: `CUDARC_CUDA_VERSION=13030`, or a `cuda-13030` feature
//! on cudarc in the user's own `Cargo.toml`.

use std::borrow::Cow;
use std::ffi::{CStr, CString};

use cudarc::nvrtc::result::{self, NvrtcError};
use cudarc::nvrtc::sys;

use crate::backend::shader::{Cubin, Cuda, Ptx};
use crate::error::Error;
use crate::tools::compile::ShaderCompiler;

/// NVRTC, with the options it compiles with.
#[derive(Debug, Clone, Default)]
pub struct Nvrtc {
    /// The target: `compute_XX` (PTX only) or `sm_XX`. `None` is NVRTC's
    /// default, PTX only.
    pub arch: Option<String>,
}

impl ShaderCompiler<Cuda, Ptx> for Nvrtc {
    fn compile(&self, input: &Cuda) -> Result<Ptx, Error> {
        let mut bytes = self.run(input, ptx)?;
        // NVRTC ends the PTX with a NUL.
        bytes.retain(|&b| b != 0);
        let ptx = String::from_utf8(bytes).map_err(|_| error("the PTX isn't UTF-8"))?;
        Ok(Ptx(Cow::Owned(ptx)))
    }
}

impl ShaderCompiler<Cuda, Cubin> for Nvrtc {
    /// Also fails unless `arch` is an `sm_XX`: a cubin is for one real
    /// architecture.
    fn compile(&self, input: &Cuda) -> Result<Cubin, Error> {
        if !self
            .arch
            .as_deref()
            .is_some_and(|arch| arch.starts_with("sm_"))
        {
            return Err(error("a cubin needs an sm_XX arch"));
        }
        Ok(Cubin(Cow::Owned(self.run(input, cubin)?)))
    }
}

impl Nvrtc {
    // Compiles `code`, then reads the output with `read`.
    fn run(
        &self,
        code: &Cuda,
        read: unsafe fn(sys::nvrtcProgram) -> Result<Vec<u8>, NvrtcError>,
    ) -> Result<Vec<u8>, Error> {
        // SAFETY: it only tries to open the library.
        if !unsafe { sys::is_culib_present() } {
            return Err(error("the NVRTC library isn't installed"));
        }
        let src = CString::new(code.0.as_bytes()).map_err(|_| error("the source holds a NUL"))?;
        let prog = result::create_program(&src, None).map_err(|e| error(e.to_string()))?;
        let options: Vec<String> = self
            .arch
            .as_deref()
            .map(|arch| format!("--gpu-architecture={arch}"))
            .into_iter()
            .collect();
        // SAFETY: `prog` was created above and is destroyed only below; `src`
        // outlives it.
        let out = unsafe {
            match result::compile_program(prog, &options) {
                Ok(()) => read(prog).map_err(|e| error(e.to_string())),
                Err(e) => Err(error(log(prog).unwrap_or_else(|| e.to_string()))),
            }
        };
        // SAFETY: as above; nothing uses `prog` after this.
        let _ = unsafe { result::destroy_program(prog) };
        out
    }
}

// SAFETY (each): `prog` is a live program that `compile_program` compiled.

unsafe fn ptx(prog: sys::nvrtcProgram) -> Result<Vec<u8>, NvrtcError> {
    let chars = unsafe { result::get_ptx(prog) }?;
    Ok(chars.into_iter().map(|c| c as u8).collect())
}

unsafe fn cubin(prog: sys::nvrtcProgram) -> Result<Vec<u8>, NvrtcError> {
    let mut size = 0;
    unsafe { sys::nvrtcGetCUBINSize(prog, &mut size) }.result()?;
    let mut bytes = vec![0u8; size];
    unsafe { sys::nvrtcGetCUBIN(prog, bytes.as_mut_ptr().cast()) }.result()?;
    Ok(bytes)
}

// The compile log, if NVRTC gives one.
unsafe fn log(prog: sys::nvrtcProgram) -> Option<String> {
    let chars = unsafe { result::get_program_log(prog) }.ok()?;
    // SAFETY: NVRTC ends the log with a NUL.
    let log = unsafe { CStr::from_ptr(chars.as_ptr()) };
    Some(log.to_string_lossy().into_owned())
}

fn error(log: impl Into<String>) -> Error {
    Error::Compile {
        compiler: "nvrtc",
        log: log.into(),
    }
}

#[cfg(test)]
#[path = "../../../tests/nvrtc.rs"]
mod tests;
