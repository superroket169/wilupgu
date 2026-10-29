use super::*;
use crate::backend::dtype::{DataType, F16, F32};
use crate::backend::Topology;
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::{ComputeDevice, DeviceConfig};
use crate::tools::builtins::{ADD, SCALE};

fn device() -> ComputeDevice<CpuBackend> {
    ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    )
}

fn n(len: u32) -> MetaSource {
    MetaSource::Uint(len.into())
}

#[test]
fn built_plan_runs_its_calls_in_order() {
    let mut p = PlanBuilder::new();
    let x = p
        .tensor(DataKind::F32, 2)
        .upload(F32::wrap(vec![1.0, 2.0]))
        .unwrap();
    let y = p.tensor(DataKind::F32, 2).zero();
    let wg = Workgroups::linear(1);
    let scale = SCALE.shader_for::<CpuBackend>().unwrap();
    let add = ADD.shader_for::<CpuBackend>().unwrap();
    p.call(scale, vec![n(2), MetaSource::Float(3.0.into())], wg, &[x])
        .unwrap();
    p.call(add, vec![n(2)], wg, &[y, x]).unwrap();

    let dev = device();
    let mut run = p.finish().unwrap().build(&dev).map_err(|(_, e)| e).unwrap();
    dev.scope(|s| run.bind(s)?.execute()).unwrap();
    let values = |id| run.tensor(id).unwrap().download_vec::<F32>().unwrap();
    assert_eq!(values(x), vec![3.0, 6.0]);
    assert_eq!(values(y), vec![3.0, 6.0]);
}

#[test]
fn what_it_holds_reads_back_in_order() {
    let mut p = PlanBuilder::new();
    let x = p.tensor(DataKind::F32, 4).blank();
    let y = p.tensor(DataKind::F16, 4).zero();
    let add = ADD.shader_for::<CpuBackend>().unwrap();
    let call = p
        .call(add, vec![n(4)], Workgroups::linear(1), &[x, x])
        .unwrap();

    let ids: Vec<_> = p.tensors().iter().map(TensorSpec::id).collect();
    assert_eq!(ids, vec![x, y]);
    assert_eq!(p.calls()[0].id(), call);
    assert_eq!(p.calls()[0].tensors(), &[x, x]);
}

#[test]
fn upload_of_another_kind_fails() {
    let mut p = PlanBuilder::new();
    let err = p
        .tensor(DataKind::F32, 2)
        .upload(F16::wrap(vec![half::f16::ONE, half::f16::ONE]));
    assert!(err.is_err());
    assert!(p.tensors().is_empty());
}

#[test]
fn call_with_a_missing_slot_fails() {
    let mut p = PlanBuilder::new();
    let y = p.tensor(DataKind::F32, 2).zero();
    let add = ADD.shader_for::<CpuBackend>().unwrap();
    assert!(p
        .call(add, vec![n(2)], Workgroups::linear(1), &[y])
        .is_err());
    assert!(p.calls().is_empty());
}

#[test]
fn finish_fails_on_a_tensor_of_another_builder() {
    let mut other = PlanBuilder::new();
    let foreign = other.tensor(DataKind::F32, 2).zero();
    let mut p = PlanBuilder::new();
    let y = p.tensor(DataKind::F32, 2).zero();
    let add = ADD.shader_for::<CpuBackend>().unwrap();
    p.call(add, vec![n(2)], Workgroups::linear(1), &[y, foreign])
        .unwrap();
    assert!(matches!(p.finish(), Err(Error::NotInPlan { .. })));
}
