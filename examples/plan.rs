use wilupgu::backend::dtype::{DataKind, DataType, F32};
use wilupgu::backend::shader::{Shader, ShaderBinary, Workgroups};
use wilupgu::backend::Topology;
use wilupgu::backends::CpuBackend;
use wilupgu::safe::{ComputeDevice, DeviceConfig};
use wilupgu::tools::builtins::{ADD, SCALE};
use wilupgu::tools::plan::call::{CallDesc, CallSpec};
use wilupgu::tools::plan::deferred::{Dynamic, Resolvable};
use wilupgu::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use wilupgu::tools::plan::run::{PlanDesc, RunPlan};
use wilupgu::tools::plan::spec::Spec;
use wilupgu::tools::plan::tensor::{TensorDesc, TensorInit, TensorSpec};
use wilupgu::Error;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpu = CpuBackend::attach(CpuBackend::choosable_devices().remove(0))?;
    let dev = ComputeDevice::new(cpu, DeviceConfig::default());

    // Described before anything exists on the device.
    let n = 3;
    let x = TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len: Resolvable::fixed(n),
        init: TensorInit::Upload(F32::wrap(vec![1.0, 2.0, 3.0])),
    })?;
    let y = TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len: Resolvable::fixed(n),
        init: TensorInit::Zero,
    })?;
    let (x_id, y_id) = (x.id(), y.id());

    let factor = Dynamic::new(); // set from the host between runs
    let wg = Resolvable::fixed(Workgroups::linear(1));
    let len = || MetaSource::Uint(Resolvable::fixed(n as u32));
    let scale = NodeSpec::new(NodeDesc {
        shader: Shader::new(&SCALE.signature, ShaderBinary::Native(SCALE.native)),
        meta: vec![len(), MetaSource::FloatPerRun(factor.clone())],
        workgroups: wg.clone(),
    })?;
    let add = NodeSpec::new(NodeDesc {
        shader: Shader::new(&ADD.signature, ShaderBinary::Native(ADD.native)),
        meta: vec![len()],
        workgroups: wg,
    })?;

    // x *= factor, then y += x.
    let plan = RunPlan::new(PlanDesc {
        tensors: vec![x, y],
        calls: vec![
            CallSpec::new(CallDesc {
                node: scale,
                tensors: vec![x_id],
            })?,
            CallSpec::new(CallDesc {
                node: add,
                tensors: vec![y_id, x_id],
            })?,
        ],
    })?;

    let mut runnable = plan.build(&dev)?;
    dev.scope(|s| {
        let bound = runnable.bind(s)?; // bound once, run as often as needed
        for f in [2.0, 3.0] {
            factor.set(f);
            bound.apply_dynamics()?;
            bound.run()?;
        }
        Ok::<_, Error>(())
    })?;

    assert_eq!(
        runnable.tensor(x_id)?.download_vec::<F32>()?,
        vec![6.0, 12.0, 18.0]
    );
    assert_eq!(
        runnable.tensor(y_id)?.download_vec::<F32>()?,
        vec![8.0, 16.0, 24.0]
    );
    Ok(())
}
