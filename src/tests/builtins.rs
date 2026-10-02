use super::*;
use crate::core::shader::ShaderFormat;

#[test]
fn all_lists_every_builtin_once() {
    assert_eq!(ALL.len(), 17);
    for (i, s) in ALL.iter().enumerate() {
        assert!(
            ALL[..i].iter().all(|o| o.name != s.name),
            "`{}` is listed twice",
            s.name
        );
    }
}

#[test]
fn every_builtin_has_some_code() {
    let formats = [ShaderFormat::Wgsl, ShaderFormat::Native, ShaderFormat::Cuda];
    for s in ALL {
        assert!(
            formats.iter().any(|&f| s.shader_code.has(f)),
            "`{}` has no code in any format",
            s.name
        );
    }
}

#[test]
fn code_paths_follow_the_shader_name() {
    let wgsl = ADD.shader_code.wgsl.unwrap();
    assert_eq!(wgsl.path(), "src/shader-codes/wgsl/add.wgsl");
    assert!(wgsl.source().contains("@compute"));
    let cuda = ADD.shader_code.cuda.unwrap();
    assert_eq!(cuda.path(), "src/shader-codes/cuda/add.cu");
}
