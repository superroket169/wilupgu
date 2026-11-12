use super::*;
use crate::backend::dtype::{DataKind, DataType, F32};
use crate::backend::shader::{Shader, Workgroups};
use crate::backends::CpuBackend;
use crate::tools::builtins::{ADD, SCALE};
use crate::tools::plan::call::CallDesc;
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use crate::tools::plan::tensor::{TensorDesc, TensorInit};

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

// x *= 2
fn double(x: GlobalId<TensorSpec>) -> CallSpec {
    let factor = MetaSource::Float(Resolvable::fixed(2.0));
    call(
        SCALE.shader_for::<CpuBackend>().unwrap(),
        vec![n(1), factor],
        vec![x],
    )
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
