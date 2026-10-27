use super::*;
use crate::backends::CpuBackend;

// Every report, or the failed ones printed in the panic.
fn assert_all_pass(reports: &[Report]) {
    let failed: Vec<String> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.to_string())
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn every_builtin_has_cases() {
    for builtin in ALL {
        for with_native in [true, false] {
            let cases = cases(builtin, with_native);
            assert!(
                !cases.is_empty(),
                "`{}` has no case",
                builtin.signature.name
            );
            for case in &cases {
                assert_eq!(case.checks().len(), builtin.signature.layout.len());
            }
        }
    }
}

#[test]
fn reductions_compare_totals_only_with_native() {
    let sum = ALL.iter().find(|b| b.signature.name == "sum").unwrap();
    assert!(matches!(
        cases(sum, true)[0].checks()[1],
        Check::Total { .. }
    ));
    assert!(matches!(
        cases(sum, false)[0].checks()[1],
        Check::Close { .. }
    ));
}

// Native against native: no backend is tested, but the table is, and the
// canary and the pad check every native shader.
#[test]
fn every_builtin_matches_itself_on_two_cpus() {
    let a = ComputeDevice::<CpuBackend>::first().unwrap();
    let b = ComputeDevice::<CpuBackend>::first().unwrap();
    assert_all_pass(&compare_builtins(&a, &b).unwrap());
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_matches_cpu() {
    use crate::backends::WgpuBackend;
    let Ok(gpu) = ComputeDevice::<WgpuBackend>::first() else {
        eprintln!("no wgpu adapter; skipped");
        return;
    };
    let cpu = ComputeDevice::<CpuBackend>::first().unwrap();
    assert_all_pass(&compare_builtins(&gpu, &cpu).unwrap());
}

#[cfg(all(feature = "cuda", feature = "compile-nvrtc"))]
#[test]
fn cuda_matches_cpu() {
    use crate::backends::CudaBackend;
    let Ok(gpu) = ComputeDevice::<CudaBackend>::first() else {
        eprintln!("no CUDA device; skipped");
        return;
    };
    let cpu = ComputeDevice::<CpuBackend>::first().unwrap();
    assert_all_pass(&compare_builtins(&gpu, &cpu).unwrap());
}

#[cfg(all(feature = "wgpu", feature = "cuda", feature = "compile-nvrtc"))]
#[test]
fn wgpu_matches_cuda() {
    use crate::backends::{CudaBackend, WgpuBackend};
    let (Ok(wgpu), Ok(cuda)) = (
        ComputeDevice::<WgpuBackend>::first(),
        ComputeDevice::<CudaBackend>::first(),
    ) else {
        eprintln!("no wgpu adapter or CUDA device; skipped");
        return;
    };
    assert_all_pass(&compare_builtins(&wgpu, &cuda).unwrap());
}
