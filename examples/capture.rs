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
    let run = p.finish()?.build(&dev).map_err(|(_, e)| e)?;

    // Recorded once, nothing runs while recording; then replayed 3 times.
    let mut captured = run.capture().map_err(|(_, e)| e)?;
    dev.scope(|s| {
        let r = captured.bind(s)?;
        (0..3).try_for_each(|_| r.replay())
    })?;
    let out = captured.tensor(x)?.download_vec::<F32>()?;
    assert_eq!(out, vec![8.0, 16.0, 24.0]);

    // New contents, same recording.
    captured.upload(x, &F32::wrap(vec![1.0, 1.0, 1.0]))?;
    dev.scope(|s| captured.bind(s)?.replay())?;
    let out = captured.tensor(x)?.download_vec::<F32>()?;
    assert_eq!(out, vec![2.0, 2.0, 2.0]);
    Ok(())
}
