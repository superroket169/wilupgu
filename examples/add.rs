use wilupgu::backend::dtype::{DataKind, DataType, F32};
use wilupgu::backend::shader::{Shader, ShaderBinary, Workgroups};
use wilupgu::backend::Topology;
use wilupgu::backends::CpuBackend;
use wilupgu::safe::{Arg, ComputeDevice, DeviceConfig, MetaValue};
use wilupgu::tools::builtins::ADD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpu = CpuBackend::attach(CpuBackend::choosable_devices().remove(0))?;
    let dev = ComputeDevice::new(cpu, DeviceConfig::default());

    let mut x = dev.alloc(DataKind::F32, 3)?;
    let mut y = dev.alloc(DataKind::F32, 3)?;
    x.upload(&F32::wrap(vec![1.0, 2.0, 3.0]))?;
    y.upload(&F32::wrap(vec![10.0, 20.0, 30.0]))?;

    // ADD is `x[i] += y[i]` for i < n; its meta is n. The CPU backend runs
    // its native code.
    let add = Shader::new(&ADD.signature, ShaderBinary::Native(ADD.native));
    let mut add = dev.node(&add, &[MetaValue::Uint(3)], Workgroups::linear(1))?;

    // Buffers and nodes are bound for the scope; it waits for the device
    // before it returns.
    dev.scope(|s| {
        let (xs, ys) = (s.bind(&mut x)?, s.bind(&mut y)?);
        let add = s.bind_node(&mut add)?;
        s.run(add, &[Arg::Write(xs), Arg::Read(ys)])
    })?;

    assert_eq!(F32::unwrap(x.download()), Some(vec![11.0, 22.0, 33.0]));
    Ok(())
}
