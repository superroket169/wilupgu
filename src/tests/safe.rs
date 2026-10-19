use super::*;
use crate::backend::dtype::{DataType, Int4, F32};
use crate::backend::shader::{MetaField, NativeBinding, NativeCode, ShaderCode};
use crate::backend::Buffer;
use crate::backends::{CpuBackend, CpuInfo};

fn scale(meta: &[u32], bindings: &mut [NativeBinding]) {
    let [NativeBinding::Write(x)] = bindings else {
        panic!("scale_like takes one written slot")
    };
    let x = bytemuck::cast_slice_mut::<u8, f32>(x);
    let factor = f32::from_bits(meta[1]);
    for v in &mut x[..meta[0] as usize] {
        *v *= factor;
    }
}

fn add(meta: &[u32], bindings: &mut [NativeBinding]) {
    let [NativeBinding::Write(acc), NativeBinding::Read(x)] = bindings else {
        panic!("add_like takes a written and a read slot")
    };
    let acc = bytemuck::cast_slice_mut::<u8, f32>(acc);
    let x = bytemuck::cast_slice::<u8, f32>(x);
    for i in 0..meta[0] as usize {
        acc[i] += x[i];
    }
}

static ADD_LIKE: Shader = Shader {
    name: "add_like",
    meta: &[MetaField {
        name: "n",
        ty: MetaType::Uint,
    }],
    layout: &[
        BindingRole::Accumulate(DataKind::F32),
        BindingRole::Input(DataKind::F32),
    ],
    min_lens: |m, _, l| l.fill(m[0] as usize),
    workgroup_size: [256, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(add)),
        ..ShaderCode::NONE
    },
};

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
    min_lens: |m, _, l| l.fill(m[0] as usize),
    workgroup_size: [256, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(scale)),
        ..ShaderCode::NONE
    },
};

static NO_CODE: Shader = Shader {
    name: "no_code",
    meta: &[],
    layout: &[],
    min_lens: |_, _, _| {},
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

fn f32s<'d>(dev: &'d ComputeDevice<CpuBackend>, values: &[f32]) -> Buf<'d, CpuBackend> {
    let mut buf = dev.alloc(DataKind::F32, values.len()).unwrap();
    buf.upload(&F32::wrap(values.to_vec())).unwrap();
    buf
}

#[test]
fn scope_chains_runs_on_the_same_buffers() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0, 2.0, 3.0]);
    let mut y = f32s(&dev, &[10.0, 20.0, 30.0]);
    let wg = Workgroups::linear(1);
    let mut add = dev.node(&ADD_LIKE, &[MetaValue::Uint(3)], wg).unwrap();
    let mut double = dev
        .node(
            &SCALE_LIKE,
            &[MetaValue::Uint(3), MetaValue::Float(2.0)],
            wg,
        )
        .unwrap();

    let inside = dev
        .scope(|s| -> Result<HostData, String> {
            let (x, y) = (s.bind(&mut x)?, s.bind(&mut y)?);
            let (add, double) = (s.bind_node(&mut add)?, s.bind_node(&mut double)?);
            s.run(add, &[Arg::Write(y), Arg::Read(x)])?;
            s.run(double, &[Arg::Write(y)])?;
            Ok(s.download(y))
        })
        .unwrap();

    assert_eq!(F32::unwrap(inside), Some(vec![22.0, 44.0, 66.0]));
    assert_eq!(F32::unwrap(y.download()), Some(vec![22.0, 44.0, 66.0]));
}

#[test]
fn scope_upload_and_update_meta_apply_to_the_next_run() {
    let dev = device(0);
    let mut x = f32s(&dev, &[0.0, 0.0]);
    let mut node = dev
        .node(
            &SCALE_LIKE,
            &[MetaValue::Uint(2), MetaValue::Float(2.0)],
            Workgroups::linear(1),
        )
        .unwrap();

    dev.scope(|s| -> Result<(), String> {
        let x = s.bind(&mut x)?;
        let node = s.bind_node(&mut node)?;
        s.upload(x, &F32::wrap(vec![1.0, 2.0]))?;
        s.run(node, &[Arg::Write(x)])?;
        s.update_meta(node, &[MetaValue::Uint(2), MetaValue::Float(10.0)])?;
        s.run(node, &[Arg::Write(x)])
    })
    .unwrap();

    assert_eq!(F32::unwrap(x.download()), Some(vec![20.0, 40.0]));
}

#[test]
fn run_rejects_wrong_args() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0]);
    let mut y = f32s(&dev, &[1.0]);
    let mut bytes = dev.alloc(DataKind::Int8, 4).unwrap();
    let mut add = dev
        .node(&ADD_LIKE, &[MetaValue::Uint(1)], Workgroups::linear(1))
        .unwrap();

    dev.scope(|s| {
        let (x, y, bytes) = (
            s.bind(&mut x).unwrap(),
            s.bind(&mut y).unwrap(),
            s.bind(&mut bytes).unwrap(),
        );
        let add = s.bind_node(&mut add).unwrap();
        // slot count
        assert!(s.run(add, &[Arg::Write(y)]).is_err());
        // Read on the Accumulate slot
        assert!(s.run(add, &[Arg::Read(y), Arg::Read(x)]).is_err());
        // Write on the Input slot
        assert!(s.run(add, &[Arg::Write(y), Arg::Write(x)]).is_err());
        // buffer kind
        assert!(s.run(add, &[Arg::Write(bytes), Arg::Read(x)]).is_err());
        // a written buffer also in another slot
        assert!(s.run(add, &[Arg::Write(y), Arg::Read(y)]).is_err());
        assert!(s.run(add, &[Arg::Write(y), Arg::Read(x)]).is_ok());
    });
}

#[test]
fn bind_rejects_another_devices_buffer_and_node() {
    let dev = device(0);
    let other = device(0);
    let mut buf = f32s(&other, &[1.0]);
    let mut node = other
        .node(&ADD_LIKE, &[MetaValue::Uint(1)], Workgroups::linear(1))
        .unwrap();

    dev.scope(|s| {
        assert!(s.bind(&mut buf).is_err());
        assert!(s.bind_node(&mut node).is_err());
    });
}
