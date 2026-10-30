use wilupgu::backends::CpuBackend;
use wilupgu::prelude::*;
use wilupgu::tools::builtins::SCALE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = ComputeDevice::<CpuBackend>::first()?;

    // x *= 2
    let mut p = PlanBuilder::new();
    let data = F32::wrap(vec![1.0, 2.0, 3.0]);
    let x = p.tensor(DataKind::F32, 3).upload(data)?;
    let scale = SCALE.shader_for::<CpuBackend>()?;
    let wg = Workgroups::cover(3, &scale);
    let meta = vec![MetaSource::Uint(3.into()), MetaSource::Float(2.0.into())];
    p.call(scale, meta, wg, &[x])?;
    let plan = p.finish()?;

    // Two runs of one plan, one x: `a` holds it, `b` only its place.
    let mut b = plan.build_hollow(&dev)?;
    let mut a = plan.build(&dev).map_err(|(_, e)| e)?;

    dev.scope(|s| a.bind(s)?.execute())?;
    b.fill_from(&mut a, x)?; // moved, not copied; now `a` is hollow
    dev.scope(|s| b.bind(s)?.execute())?;

    assert!(a.tensor(x).is_err());
    assert_eq!(b.tensor(x)?.download_vec::<F32>()?, vec![4.0, 8.0, 12.0]);
    Ok(())
}
