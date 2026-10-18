use super::*;
use crate::backend::dtype::{DataType, Int4, F32};
use crate::backend::shader::{MetaField, NativeBinding, NativeCode, ShaderCode};
use crate::backend::Buffer;
use crate::backends::{CpuBackend, CpuInfo};

fn noop(_meta: &[u32], _bindings: &mut [NativeBinding]) {}

static SCALE_LIKE: Shader = Shader {
    name: "scale_like",
    meta: &[
        MetaField {
            name: "n",
            ty: MetaType::Uint,
        },
        MetaField {
            name: "factor",
            ty: MetaType::Float,
        },
    ],
    layout: &[BindingRole::InOut(DataKind::F32)],
    workgroup_size: [256, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(noop)),
        ..ShaderCode::NONE
    },
};

static NO_CODE: Shader = Shader {
    name: "no_code",
    meta: &[],
    layout: &[],
    workgroup_size: [1, 1, 1],
    shader_code: ShaderCode::NONE,
};

fn device(pool_bytes: u64) -> ComputeDevice<CpuBackend> {
    ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig { pool_bytes },
    )
}

#[test]
fn alloc_rounds_int4_up_to_whole_words() {
    let dev = device(0);
    for (asked, got) in [(0, 0), (1, 8), (8, 8), (9, 16), (13, 16)] {
        assert_eq!(dev.alloc(DataKind::Int4, asked).unwrap().len(), got);
    }
    assert_eq!(dev.alloc(DataKind::F32, 13).unwrap().len(), 13);
}

#[test]
fn download_returns_what_was_uploaded() {
    let dev = device(0);
    let mut buf = dev.alloc(DataKind::F32, 3).unwrap();
    buf.upload(&F32::wrap(vec![1.0, 2.0, 3.0])).unwrap();
    assert_eq!(F32::unwrap(buf.download()), Some(vec![1.0, 2.0, 3.0]));

    let mut packed = dev.alloc(DataKind::Int4, 13).unwrap();
    packed
        .upload(&Int4::wrap(vec![0x7654_3210, 0xfedc_ba98]))
        .unwrap();
    assert_eq!(
        Int4::unwrap(packed.download()),
        Some(vec![0x7654_3210, 0xfedc_ba98])
    );
}

#[test]
fn upload_rejects_another_kind_or_length() {
    let dev = device(0);
    let mut buf = dev.alloc(DataKind::F32, 2).unwrap();
    assert!(buf.upload(&F32::wrap(vec![1.0])).is_err());
    assert!(buf.upload(&F32::wrap(vec![1.0; 3])).is_err());
    assert!(buf.upload(&Int4::wrap(vec![0])).is_err());
}

#[test]
fn dropped_buf_goes_back_to_the_pool() {
    let dev = device(1024);
    let buf = dev.alloc(DataKind::F32, 4).unwrap();
    let addr = buf.raw().address();
    drop(buf);
    assert_eq!(dev.pool.free_bytes(), 16);
    assert_eq!(dev.alloc(DataKind::F32, 4).unwrap().raw().address(), addr);
    assert_eq!(dev.pool.free_bytes(), 16);
}

#[test]
fn pool_off_keeps_nothing() {
    let dev = device(0);
    drop(dev.alloc(DataKind::F32, 4).unwrap());
    assert_eq!(dev.pool.free_bytes(), 0);
}

#[test]
fn node_checks_meta_count_and_types() {
    let dev = device(0);
    let wg = Workgroups::linear(1);
    let good = [MetaValue::Uint(4), MetaValue::Float(2.0)];
    assert!(dev.node(&SCALE_LIKE, &good, wg).is_ok());
    assert!(dev.node(&SCALE_LIKE, &good[..1], wg).is_err());
    assert!(dev
        .node(&SCALE_LIKE, &[MetaValue::Uint(4), MetaValue::Uint(2)], wg)
        .is_err());
}

#[test]
fn node_needs_code_in_the_backends_format() {
    let dev = device(0);
    assert!(dev.node(&NO_CODE, &[], Workgroups::linear(1)).is_err());
}

#[test]
fn update_meta_checks_like_node() {
    let dev = device(0);
    let good = [MetaValue::Uint(4), MetaValue::Float(2.0)];
    let mut node = dev.node(&SCALE_LIKE, &good, Workgroups::linear(1)).unwrap();
    assert!(node
        .update_meta(&[MetaValue::Uint(8), MetaValue::Float(0.5)])
        .is_ok());
    assert!(node
        .update_meta(&[MetaValue::Float(8.0), MetaValue::Float(0.5)])
        .is_err());
}
