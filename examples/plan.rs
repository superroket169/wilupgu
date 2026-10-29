use wilupgu::backends::CpuBackend;
use wilupgu::prelude::*;
use wilupgu::tools::builtins::{ADD, SCALE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = ComputeDevice::<CpuBackend>::first()?;

    // Described before anything exists on the device.
    let n: u32 = 3;
    let mut p = PlanBuilder::new();
    let x = p
        .tensor(DataKind::F32, n as usize)
        .upload(F32::wrap(vec![1.0, 2.0, 3.0]))?;
    let y = p.tensor(DataKind::F32, n as usize).zero();

    // x *= factor, then y += x.
    let factor = Dynamic::new(); // set from the host between runs
    let len = || MetaSource::Uint(n.into());
    let (scale, add) = (
        SCALE.shader_for::<CpuBackend>()?,
        ADD.shader_for::<CpuBackend>()?,
    );
    let per_run_factor = MetaSource::FloatPerRun(factor.clone());
    let (scale_wg, add_wg) = (Workgroups::cover(n, &scale), Workgroups::cover(n, &add));
    p.call(scale, vec![len(), per_run_factor], scale_wg, &[x])?;
    p.call(add, vec![len()], add_wg, &[y, x])?;

    let mut run = p.finish()?.build(&dev).map_err(|(_, e)| e)?;
    dev.scope(|s| {
        let bound = run.bind(s)?; // bound once, run as often as needed
        for f in [2.0, 3.0] {
            factor.set(f);
            bound.apply_dynamics()?;
            bound.execute()?;
        }
        Ok::<_, Error>(())
    })?;

    assert_eq!(run.tensor(x)?.download_vec::<F32>()?, vec![6.0, 12.0, 18.0]);
    assert_eq!(run.tensor(y)?.download_vec::<F32>()?, vec![8.0, 16.0, 24.0]);
    Ok(())
}
