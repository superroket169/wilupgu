use wilupgu::backends::CpuBackend;
use wilupgu::prelude::*;
use wilupgu::tools::parity::builtins::compare_builtins;

// Every builtin on `A` and on `B`: one line per passing case, the full
// report for a failing one. `None` if either backend has no device here.
fn pair<A: Backend + Topology, B: Backend + Topology>() -> Result<Option<(usize, usize)>, Error> {
    let (Ok(a), Ok(b)) = (ComputeDevice::<A>::first(), ComputeDevice::<B>::first()) else {
        println!("{} vs {}: no device, skipped\n", A::NAME, B::NAME);
        return Ok(None);
    };
    println!("{} vs {}", A::NAME, B::NAME);
    let reports = compare_builtins(&a, &b)?;
    let passed = reports.iter().filter(|r| r.passed()).count();
    for r in &reports {
        if r.passed() {
            println!("  ok    {}  {:?}", r.shader(), r.case().meta());
        } else {
            print!("{r}");
        }
    }
    println!("  {passed}/{} passed\n", reports.len());
    Ok(Some((passed, reports.len())))
}

fn main() -> Result<(), Error> {
    let mut runs = vec![pair::<CpuBackend, CpuBackend>()?];
    #[cfg(feature = "wgpu")]
    runs.push(pair::<wilupgu::backends::WgpuBackend, CpuBackend>()?);
    #[cfg(all(feature = "cuda", feature = "compile-nvrtc"))]
    runs.push(pair::<wilupgu::backends::CudaBackend, CpuBackend>()?);
    #[cfg(all(feature = "wgpu", feature = "cuda", feature = "compile-nvrtc"))]
    runs.push(pair::<
        wilupgu::backends::WgpuBackend,
        wilupgu::backends::CudaBackend,
    >()?);

    if runs
        .into_iter()
        .flatten()
        .any(|(passed, all)| passed != all)
    {
        std::process::exit(1);
    }
    Ok(())
}
