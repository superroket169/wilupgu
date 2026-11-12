use super::*;
use crate::backend::dtype::{DataKind, DataType, F32};
use crate::backend::shader::{Shader, Workgroups};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::DeviceConfig;
use crate::tools::builtins::{ADD, SCALE};
use crate::tools::plan::call::CallDesc;
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use crate::tools::plan::tensor::{TensorDesc, TensorInit};

fn device() -> ComputeDevice<CpuBackend> {
    ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    )
}

fn f32_tensor(data: Vec<f32>) -> TensorSpec {
    TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len: Resolvable::fixed(data.len()),
        init: TensorInit::Upload(F32::wrap(data)),
    })
    .unwrap()
}

fn call(shader: Shader, meta: Vec<MetaSource>, tensors: Vec<GlobalId<TensorSpec>>) -> CallSpec {
    let workgroups = Resolvable::fixed(Workgroups { x: 1, y: 1, z: 1 });
    let node = NodeSpec::new(NodeDesc {
        shader,
        meta,
        workgroups,
    })
    .unwrap();
    CallSpec::new(CallDesc { node, tensors }).unwrap()
}

fn n(v: u32) -> MetaSource {
    MetaSource::Uint(Resolvable::fixed(v))
}

// x *= 2, on the first `len` elements
fn double_n(x: GlobalId<TensorSpec>, len: u32) -> CallSpec {
    let factor = MetaSource::Float(Resolvable::fixed(2.0));
    call(
        SCALE.shader_for::<CpuBackend>().unwrap(),
        vec![n(len), factor],
        vec![x],
    )
}

fn double(x: GlobalId<TensorSpec>) -> CallSpec {
    double_n(x, 1)
}

// acc += x
fn add(acc: GlobalId<TensorSpec>, x: GlobalId<TensorSpec>) -> CallSpec {
    call(
        ADD.shader_for::<CpuBackend>().unwrap(),
        vec![n(1)],
        vec![acc, x],
    )
}

#[test]
fn routines_share_the_mesh_tensors() {
    let (x, y) = (f32_tensor(vec![1.0]), f32_tensor(vec![0.0]));
    let (xi, yi) = (x.id(), y.id());
    let (first, second) = (Routine::key(), Routine::key());
    let mesh = MeshSpec::new(MeshDesc {
        tensors: vec![x, y],
        routines: vec![(first, vec![double(xi)]), (second, vec![add(yi, xi)])],
    })
    .unwrap();
    assert_eq!(mesh.routine(first).unwrap().len(), 1);
    assert_eq!(mesh.routine(second).unwrap()[0].tensors(), &[yi, xi]);
    assert_eq!(
        mesh.routine(Routine::key()).err(),
        Some(Error::NotInPlan { what: "routine" })
    );
}

#[test]
fn new_rejects_a_key_given_twice() {
    let x = f32_tensor(vec![1.0]);
    let xi = x.id();
    let key = Routine::key();
    let err = MeshSpec::new(MeshDesc {
        tensors: vec![x],
        routines: vec![(key, vec![double(xi)]), (key, vec![double(xi)])],
    })
    .err();
    assert_eq!(err, Some(Error::DuplicateId { what: "routine" }));
}

#[test]
fn new_rejects_a_tensor_outside_the_mesh() {
    let x = f32_tensor(vec![1.0]);
    let err = MeshSpec::new(MeshDesc {
        tensors: vec![],
        routines: vec![(Routine::key(), vec![double(x.id())])],
    })
    .err();
    assert_eq!(err, Some(Error::NotInPlan { what: "tensor" }));
}

#[test]
fn build_makes_every_tensor_once_and_every_routine_call() {
    let (x, y) = (f32_tensor(vec![1.0]), f32_tensor(vec![5.0]));
    let (xi, yi) = (x.id(), y.id());
    let (first, second) = (Routine::key(), Routine::key());
    let spec = MeshSpec::new(MeshDesc {
        tensors: vec![x, y],
        routines: vec![
            (first, vec![double(xi)]),
            (second, vec![add(yi, xi), double(yi)]),
        ],
    })
    .unwrap();
    let dev = device();
    let mesh = spec.build(&dev).map_err(|(_, e)| e).unwrap();
    assert_eq!(mesh.tensors().count(), 2);
    assert_eq!(
        mesh.tensor(yi).unwrap().download_vec::<F32>(),
        Ok(vec![5.0])
    );
    let calls = mesh.routine(second).unwrap();
    assert_eq!(calls[0].tensors(), &[yi, xi]);
    assert_eq!(calls[1].tensors(), &[yi]);
    assert_eq!(mesh.routines().count(), 2);
}

#[test]
fn build_rejects_a_tensor_shorter_than_its_slot_needs() {
    let x = f32_tensor(vec![1.0]);
    let xi = x.id();
    let spec = MeshSpec::new(MeshDesc {
        tensors: vec![x],
        routines: vec![(Routine::key(), vec![double_n(xi, 2)])],
    })
    .unwrap();
    let dev = device();
    let Err((_, e)) = spec.build(&dev) else {
        panic!("a call needs 2 elements of a tensor that holds 1")
    };
    assert!(matches!(e, Error::TooShort { .. }));
}

#[test]
fn build_waits_until_the_spec_is_ready() {
    let len = Resolvable::new();
    let resolver = len.resolver().unwrap();
    let x = TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len,
        init: TensorInit::Zero,
    })
    .unwrap();
    let xi = x.id();
    let spec = MeshSpec::new(MeshDesc {
        tensors: vec![x],
        routines: vec![(Routine::key(), vec![double(xi)])],
    })
    .unwrap();
    let dev = device();
    let Err((spec, _)) = spec.build(&dev) else {
        panic!("a mesh with an unknown length built")
    };
    resolver.resolve(1);
    assert!(spec.build(&dev).is_ok());
}
