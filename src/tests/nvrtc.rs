use super::*;
use crate::tools::builtins::ALL;

// SAFETY: it only tries to open the library.
fn nvrtc_present() -> bool {
    let present = unsafe { sys::is_culib_present() };
    if !present {
        eprintln!("NVRTC isn't installed; skipped");
    }
    present
}

#[test]
fn a_cubin_needs_an_sm_arch() {
    let code = Cuda(Cow::Borrowed(""));
    for arch in [None, Some("compute_75".to_string())] {
        let result: Result<Cubin, Error> = Nvrtc { arch }.compile(&code);
        assert!(matches!(result, Err(Error::Compile { .. })));
    }
}

#[test]
fn every_builtin_compiles_to_ptx() {
    if !nvrtc_present() {
        return;
    }
    for b in ALL {
        let ptx: Ptx = Nvrtc::default()
            .compile(&b.cuda)
            .unwrap_or_else(|e| panic!("`{}`: {e}", b.signature.name));
        assert!(ptx.0.contains(".entry entry"), "`{}`", b.signature.name);
    }
}

#[test]
fn a_failed_compile_gives_the_log() {
    if !nvrtc_present() {
        return;
    }
    let code = Cuda(Cow::Borrowed(
        "extern \"C\" __global__ void entry() { nope; }",
    ));
    let result: Result<Ptx, Error> = Nvrtc::default().compile(&code);
    let Err(Error::Compile { log, .. }) = result else {
        panic!("a broken source compiled");
    };
    assert!(log.contains("nope"), "{log}");
}
