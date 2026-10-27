use wilupgu::backends::CpuBackend;
use wilupgu::prelude::*;
use wilupgu::tools::builtins::ADD;

// The same code on any backend: `ADD.shader_for` picks the format it runs.
fn add<B: Backend + Topology>() -> Result<Vec<f32>, Error> {
    let dev = ComputeDevice::<B>::first()?;
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0]))?;
    let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 20.0, 30.0]))?;

    // x[i] += y[i] for i < n; the meta is n.
    let mut add = dev.node(
        &ADD.shader_for::<B>()?,
        &[MetaValue::Uint(3)],
        Workgroups::linear(1),
    )?;

    // While bound, the host can't touch x or y; the scope waits for the
    // device before it returns.
    dev.scope(|s| {
        let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
        s.run(add, &[Arg::Write(x), Arg::Read(y)])
    })?;
    x.download_vec::<F32>()
}

fn main() -> Result<(), Error> {
    assert_eq!(add::<CpuBackend>()?, vec![11.0, 22.0, 33.0]);
    #[cfg(feature = "wgpu")]
    println!("wgpu: {:?}", add::<wilupgu::backends::WgpuBackend>()?);
    Ok(())
}
