use super::*;

#[test]
fn wgsl_code_accepts_a_wgsl_path() {
    let code = WgslCode::new("wgsl/add.wgsl", "src");
    assert_eq!(code.path(), "wgsl/add.wgsl");
    assert_eq!(code.source(), "src");
}

#[test]
#[should_panic(expected = ".wgsl file")]
fn wgsl_code_rejects_another_extension() {
    WgslCode::new("cuda/add.cu", "");
}

#[test]
#[should_panic(expected = ".cu file")]
fn cuda_code_rejects_another_extension() {
    CudaCode::new("wgsl/add.wgsl", "");
}

#[test]
#[should_panic(expected = ".cu file")]
fn cuda_code_rejects_a_bare_extension_lookalike() {
    CudaCode::new("cu", "");
}

#[test]
fn has_reports_each_format_on_its_own() {
    let code = ShaderCode {
        cuda: Some(CudaCode::new("cuda/add.cu", "")),
        ..ShaderCode::NONE
    };
    assert!(code.has(ShaderFormat::Cuda));
    assert!(!code.has(ShaderFormat::Wgsl));
    assert!(!code.has(ShaderFormat::Native));
}
