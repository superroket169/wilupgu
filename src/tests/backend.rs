use super::*;
pub(crate) use crate::backends::toy::{ToyBackend, ToyNode};
use crate::core::node::NodeSpec;
use crate::core::shader::{BindingRole, CpuBinding, MetaField, MetaType, NativeCode, ShaderCode};
use crate::core::tensor::TensorId;
use std::sync::Arc;

pub(crate) static TOY_SHADER: Shader = Shader {
    name: "Toy",
    meta: &[],
    layout: &[],
    workgroup_size: [1, 1, 1],
    shader_code: ShaderCode::NONE,
};

fn noop(_: &[CpuBinding]) {}

pub(crate) static COPY_SHADER: Shader = Shader {
    name: "Copy",
    meta: &[],
    layout: &[
        BindingRole::Input(DataKind::F32),
        BindingRole::Output(DataKind::F32),
    ],
    workgroup_size: [1, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(noop)),
        ..ShaderCode::NONE
    },
};

pub(crate) static META_SHADER: Shader = Shader {
    name: "MetaEcho",
    meta: &[
        MetaField {
            name: "n",
            ty: MetaType::Uint,
        },
        MetaField {
            name: "lr",
            ty: MetaType::Float,
        },
    ],
    layout: &[],
    workgroup_size: [1, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(noop)),
        ..ShaderCode::NONE
    },
};

#[test]
fn topology_enumerate_then_attach() {
    let devices = ToyBackend::choosable_devices();
    assert_eq!(devices.len(), 2);
    let backend = ToyBackend::attach(devices[0].clone()).unwrap();
    assert_eq!(backend.name(), "toy");
}

#[test]
fn carve_from_area() {
    let ctx = ToyBackend::new();
    let mut area = ctx.reserve(1024);
    let buf = ctx.carve(&mut area, 4);
    assert_eq!(ctx.download(&buf, 4).len(), 4);
    ctx.release(area);
}

pub(crate) fn copy_node(from: TensorId, to: TensorId) -> NodeSpec {
    NodeSpec::new(
        &COPY_SHADER,
        vec![],
        vec![
            Binding::new(1, from, BindingRole::Input(DataKind::F32)),
            Binding::new(2, to, BindingRole::Output(DataKind::F32)),
        ],
        Workgroups::linear(1),
    )
}

// A toy device with `n` fresh one-element F32 tensors in its table.
pub(crate) fn device_with(n: usize) -> (Arc<ToyBackend>, Vec<TensorId>) {
    let toy = ToyBackend::new();
    let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();
    for &id in &ids {
        toy.table().alloc(&toy, id, DataKind::F32, 1).unwrap();
    }
    (Arc::new(toy), ids)
}

#[test]
fn supports_p2p_defaults_to_false() {
    let ctx = ToyBackend::new();
    assert!(!ctx.supports_p2p(DeviceId::new()));
}

#[test]
fn copy_to_round_trips_through_host() {
    let src = ToyBackend::new();
    let dest = ToyBackend::new();
    let buf = src.alloc(4);
    src.upload(&buf, &[1.0, 2.0, 3.0, 4.0]);

    let copy = src.copy_to(&buf, 4, &dest);

    assert_eq!(dest.download(&copy, 4), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        src.download(&buf, 4),
        vec![1.0, 2.0, 3.0, 4.0],
        "the source stays"
    );
}

#[test]
fn alloc_kind_uses_the_matching_impl() {
    let ctx = ToyBackend::new();
    let buf = ctx.alloc_kind(DataKind::F32, 4).unwrap();
    assert_eq!(buf.size_bytes(), 16);
}

#[test]
fn alloc_kind_rejects_an_unsupported_kind() {
    let err = ToyBackend::new()
        .alloc_kind(DataKind::Int4, 4)
        .err()
        .unwrap();
    assert!(err.contains("Int4"), "{err}");
}

#[test]
fn host_data_round_trips_through_the_kind_api() {
    let ctx = ToyBackend::new();
    let buf = ctx.alloc_kind(DataKind::F32, 3).unwrap();
    ctx.upload_kind(&buf, &HostData::F32(vec![1.0, 2.0, 3.0]))
        .unwrap();
    let back = ctx.download_kind(&buf, DataKind::F32, 3).unwrap();
    assert!(matches!(back, HostData::F32(v) if v == vec![1.0, 2.0, 3.0]));
}

#[test]
fn download_returns_only_the_asked_elements() {
    let ctx = ToyBackend::new();
    let buf = ctx.alloc(4);
    ctx.upload(&buf, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ctx.download(&buf, 2), vec![1.0, 2.0]);
}

#[test]
fn upload_kind_rejects_an_unsupported_kind() {
    let ctx = ToyBackend::new();
    let buf = ctx.alloc_kind(DataKind::F32, 1).unwrap();
    let err = ctx
        .upload_kind(&buf, &HostData::F16(vec![half::f16::ZERO]))
        .unwrap_err();
    assert!(err.contains("F16"), "{err}");
}
