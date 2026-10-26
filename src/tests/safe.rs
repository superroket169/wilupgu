use super::*;
use crate::backend::dtype::{DataType, Int4, Int8, F32};
use crate::backend::shader::{BindingRole, MetaField, NativeBinding, NativeCode, ShaderCode};
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

fn f32s<'d>(dev: &'d ComputeDevice<CpuBackend>, values: &[f32]) -> Tensor<'d, CpuBackend> {
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
        .scope(|s| -> Result<HostData, Error> {
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

    dev.scope(|s| -> Result<(), Error> {
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

#[test]
fn run_rejects_a_buffer_shorter_than_min_lens() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0, 2.0, 3.0]);
    let mut node = dev
        .node(
            &SCALE_LIKE,
            &[MetaValue::Uint(3), MetaValue::Float(2.0)],
            Workgroups::linear(1),
        )
        .unwrap();

    dev.scope(|s| {
        let x = s.bind(&mut x).unwrap();
        let node = s.bind_node(&mut node).unwrap();
        // n = len, n < len: fine
        assert!(s.run(node, &[Arg::Write(x)]).is_ok());
        s.update_meta(node, &[MetaValue::Uint(2), MetaValue::Float(2.0)])
            .unwrap();
        assert!(s.run(node, &[Arg::Write(x)]).is_ok());
        // n = len + 1: the buffer is one element short
        s.update_meta(node, &[MetaValue::Uint(4), MetaValue::Float(2.0)])
            .unwrap();
        assert!(s.run(node, &[Arg::Write(x)]).is_err());
    });
    // the first run doubled all three, the second the first two, the third didn't run
    assert_eq!(F32::unwrap(x.download()), Some(vec![4.0, 8.0, 6.0]));
}

#[test]
fn area_bufs_work_like_alloced_ones() {
    let dev = device(0);
    let area = dev.reserve(64);
    let mut x = area.alloc(DataKind::F32, 3).unwrap();
    let mut y = area.alloc(DataKind::F32, 3).unwrap();
    x.upload(&F32::wrap(vec![1.0, 2.0, 3.0])).unwrap();
    y.upload(&F32::wrap(vec![10.0, 20.0, 30.0])).unwrap();
    let mut add = dev
        .node(&ADD_LIKE, &[MetaValue::Uint(3)], Workgroups::linear(1))
        .unwrap();

    dev.scope(|s| -> Result<(), Error> {
        let (x, y) = (s.bind(&mut x)?, s.bind(&mut y)?);
        let add = s.bind_node(&mut add)?;
        s.run(add, &[Arg::Write(y), Arg::Read(x)])
    })
    .unwrap();

    assert_eq!(F32::unwrap(y.download()), Some(vec![11.0, 22.0, 33.0]));
}

#[test]
fn area_alloc_rounds_int4_and_fails_when_full() {
    let dev = device(0);
    let area = dev.reserve(16);
    assert_eq!(area.alloc(DataKind::Int4, 13).unwrap().len(), 16);
    assert_eq!(area.remaining_bytes(), 8);
    assert!(area.alloc(DataKind::F32, 3).is_err());
    assert_eq!(area.remaining_bytes(), 8, "a failed alloc takes nothing");
    area.alloc(DataKind::F32, 2).unwrap();
    assert_eq!(area.remaining_bytes(), 0);
}

#[test]
fn dropped_area_buf_stays_out_of_the_pool() {
    let dev = device(1024);
    let area = dev.reserve(16);
    drop(area.alloc(DataKind::F32, 4).unwrap());
    assert_eq!(dev.pool.free_bytes(), 0);
}

fn scale_node(dev: &ComputeDevice<CpuBackend>, n: u32, factor: f32) -> Node<'_, CpuBackend> {
    dev.node(
        &SCALE_LIKE,
        &[MetaValue::Uint(n), MetaValue::Float(factor)],
        Workgroups::linear(1),
    )
    .unwrap()
}

#[test]
fn capture_records_and_replay_runs_with_the_current_meta() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0, 2.0]);
    let mut y = f32s(&dev, &[3.0]);
    let mut add = dev
        .node(&ADD_LIKE, &[MetaValue::Uint(1)], Workgroups::linear(1))
        .unwrap();
    let mut double = scale_node(&dev, 2, 2.0);

    let rec = dev
        .scope(|s| {
            let (x, y) = (s.bind(&mut x)?, s.bind(&mut y)?);
            let (add, double) = (s.bind_node(&mut add)?, s.bind_node(&mut double)?);
            s.capture(|| {
                s.run(double, &[Arg::Write(x)])?;
                s.run(add, &[Arg::Write(y), Arg::Read(x)])
            })
        })
        .unwrap();
    assert_eq!(
        F32::unwrap(x.download()),
        Some(vec![1.0, 2.0]),
        "capture ran nothing"
    );

    for factor in [2.0, 10.0] {
        double
            .update_meta(&[MetaValue::Uint(2), MetaValue::Float(factor)])
            .unwrap();
        // bound in another order than the recording: buffers are found by address
        dev.scope(|s| {
            s.bind_node(&mut add)?;
            s.bind(&mut y)?;
            s.bind_node(&mut double)?;
            s.bind(&mut x)?;
            s.replay(&rec)
        })
        .unwrap();
    }
    // x: 1, 2 -> 2, 4 -> 20, 40; y: 3 + 2 + 20
    assert_eq!(F32::unwrap(x.download()), Some(vec![20.0, 40.0]));
    assert_eq!(F32::unwrap(y.download()), Some(vec![25.0]));
}

#[test]
fn replay_needs_every_recorded_buffer_and_node_bound() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0]);
    let mut node = scale_node(&dev, 1, 2.0);
    let rec = dev
        .scope(|s| {
            let (x, n) = (s.bind(&mut x)?, s.bind_node(&mut node)?);
            s.capture(|| s.run(n, &[Arg::Write(x)]))
        })
        .unwrap();

    let no_buffer = dev.scope(|s| {
        s.bind_node(&mut node)?;
        s.replay(&rec)
    });
    assert!(no_buffer.is_err());
    let no_node = dev.scope(|s| {
        s.bind(&mut x)?;
        s.replay(&rec)
    });
    assert!(no_node.is_err());

    // a moved node is another node to the recording
    let mut moved = Box::new(node);
    let moved_node = dev.scope(|s| {
        s.bind(&mut x)?;
        s.bind_node(&mut moved)?;
        s.replay(&rec)
    });
    assert!(moved_node.is_err());
    assert_eq!(F32::unwrap(x.download()), Some(vec![1.0]), "no replay ran");
}

#[test]
fn replay_checks_min_lens_again() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0, 2.0]);
    let mut node = scale_node(&dev, 2, 2.0);
    let rec = dev
        .scope(|s| {
            let (x, n) = (s.bind(&mut x)?, s.bind_node(&mut node)?);
            s.capture(|| s.run(n, &[Arg::Write(x)]))
        })
        .unwrap();

    node.update_meta(&[MetaValue::Uint(3), MetaValue::Float(2.0)])
        .unwrap();
    let too_long = dev.scope(|s| {
        s.bind(&mut x)?;
        s.bind_node(&mut node)?;
        s.replay(&rec)
    });
    assert!(too_long.is_err());
    assert_eq!(F32::unwrap(x.download()), Some(vec![1.0, 2.0]));
}

#[test]
fn a_failed_or_panicking_capture_leaves_the_device_able_to_capture() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0]);
    let mut node = scale_node(&dev, 1, 2.0);

    let failed = dev.scope(|s| {
        let (x, n) = (s.bind(&mut x)?, s.bind_node(&mut node)?);
        s.capture(|| {
            s.run(n, &[Arg::Write(x)])?;
            Err(Error::Backend("stop".to_string()))
        })
    });
    assert!(failed.is_err());

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dev.scope(|s| s.capture(|| panic!("inside capture")))
    }));
    assert!(panicked.is_err());

    let again = dev.scope(|s| {
        let (x, n) = (s.bind(&mut x)?, s.bind_node(&mut node)?);
        s.capture(|| s.run(n, &[Arg::Write(x)]))
    });
    assert!(again.is_ok());
}

#[test]
fn capture_and_replay_reject_misuse() {
    let dev = device(0);
    let other = device(0);
    let mut x = f32s(&dev, &[1.0]);
    let mut node = scale_node(&dev, 1, 2.0);
    let rec = dev
        .scope(|s| {
            let (x, n) = (s.bind(&mut x)?, s.bind_node(&mut node)?);
            s.capture(|| s.run(n, &[Arg::Write(x)]))
        })
        .unwrap();

    dev.scope(|s| {
        // nested capture, and replay while capturing
        let nested = s.capture(|| s.capture(|| Ok(())).map(drop));
        assert!(nested.is_err());
        let replay_inside = s.capture(|| s.replay(&rec));
        assert!(replay_inside.is_err());
    });
    // another device's recording
    assert!(other.scope(|s| s.replay(&rec)).is_err());
}

#[test]
fn odd_byte_lengths_round_trip() {
    use crate::backend::dtype::F16;

    let dev = device(0);
    // 3 F16s are 6 bytes, 5 Int8s are 5: neither is a multiple of COPY_ALIGN
    let mut halves = dev.alloc(DataKind::F16, 3).unwrap();
    let values: Vec<half::f16> = [1.0f32, 2.0, 3.0].map(half::f16::from_f32).to_vec();
    halves.upload(&F16::wrap(values.clone())).unwrap();
    assert_eq!(F16::unwrap(halves.download()), Some(values));

    let mut bytes = dev.alloc(DataKind::Int8, 5).unwrap();
    bytes.upload(&Int8::wrap(vec![1, 2, 3, 4, 5])).unwrap();
    assert_eq!(Int8::unwrap(bytes.download()), Some(vec![1, 2, 3, 4, 5]));
}

// --- helpers ---

#[test]
fn tensor_from_and_typed_access() {
    let dev = device(0);
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
    assert_eq!(x.download_vec::<F32>(), Ok(vec![1.0, 2.0]));
    x.upload_slice::<F32>(&[3.0, 4.0]).unwrap();
    assert_eq!(x.download_vec::<F32>(), Ok(vec![3.0, 4.0]));
    assert!(x.download_vec::<Int4>().is_err(), "another kind");
    assert!(x.upload_slice::<Int4>(&[0]).is_err(), "another kind");
    assert!(x.upload_slice::<F32>(&[1.0]).is_err(), "another length");
}

#[test]
fn copy_to_makes_an_independent_copy_on_another_device() {
    let (dev, other) = (device(0), device(0));
    let mut x = f32s(&dev, &[1.0, 2.0]);
    let copy = x.copy_to(&other).unwrap();
    x.upload_slice::<F32>(&[9.0, 9.0]).unwrap();
    assert_eq!(copy.download_vec::<F32>(), Ok(vec![1.0, 2.0]));
}

#[test]
fn upload_at_and_download_range_touch_only_their_range() {
    let dev = device(0);
    let mut x = f32s(&dev, &[0.0; 4]);
    x.upload_at(1, &F32::wrap(vec![5.0, 6.0])).unwrap();
    assert_eq!(x.download_vec::<F32>(), Ok(vec![0.0, 5.0, 6.0, 0.0]));
    assert_eq!(
        F32::unwrap(x.download_range(1, 2).unwrap()),
        Some(vec![5.0, 6.0])
    );
    assert!(
        x.upload_at(3, &F32::wrap(vec![1.0, 1.0])).is_err(),
        "past the end"
    );
    assert!(x.download_range(4, 1).is_err(), "past the end");
    assert!(
        x.upload_at(0, &Int8::wrap(vec![1; 4])).is_err(),
        "another kind"
    );
}

#[test]
fn ranges_follow_copy_align() {
    use crate::backend::dtype::F16;
    let h = |v: f32| half::f16::from_f32(v);

    let dev = device(0);
    let mut x = dev
        .tensor_from(&F16::wrap([1.0, 2.0, 3.0, 4.0, 5.0].map(h).to_vec()))
        .unwrap();
    // element 1 of F16 starts at byte 2
    assert!(x.upload_at(1, &F16::wrap(vec![h(0.0)])).is_err());
    assert!(x.download_range(1, 1).is_err());
    // starts at byte 4, but ends at byte 6, inside the tensor
    assert!(x.upload_at(2, &F16::wrap(vec![h(0.0)])).is_err());
    // a read may end anywhere
    assert_eq!(
        F16::unwrap(x.download_range(2, 1).unwrap()),
        Some(vec![h(3.0)])
    );
    // ends at the tensor's end: the rounding is padding, nothing else moves
    x.upload_at(4, &F16::wrap(vec![h(50.0)])).unwrap();
    assert_eq!(
        F16::unwrap(x.download()),
        Some([1.0, 2.0, 3.0, 4.0, 50.0].map(h).to_vec())
    );

    let packed = dev.alloc(DataKind::Int4, 16).unwrap();
    // 4 Int4s are 2 bytes, 8 are 4
    assert!(packed.download_range(4, 4).is_err());
    assert!(packed.download_range(8, 8).is_ok());
}

#[test]
fn copy_into_copies_a_range_on_the_device() {
    let dev = device(0);
    let src = f32s(&dev, &[1.0, 2.0, 3.0, 4.0]);
    let mut dst = f32s(&dev, &[0.0; 4]);
    src.copy_into(1, &mut dst, 2, 2).unwrap();
    assert_eq!(dst.download_vec::<F32>(), Ok(vec![0.0, 0.0, 2.0, 3.0]));

    let other = device(0);
    let mut far = f32s(&other, &[0.0; 4]);
    assert!(src.copy_into(0, &mut far, 0, 1).is_err(), "another device");
    let mut bytes = dev.alloc(DataKind::Int8, 16).unwrap();
    assert!(src.copy_into(0, &mut bytes, 0, 1).is_err(), "another kind");
    assert!(src.copy_into(3, &mut dst, 0, 2).is_err(), "past src's end");
}

#[test]
fn zero_clears_the_tensor() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0, 2.0, 3.0]);
    x.zero().unwrap();
    assert_eq!(x.download_vec::<F32>(), Ok(vec![0.0; 3]));
}

#[test]
fn scope_copy_lands_between_runs() {
    let dev = device(0);
    let mut x = f32s(&dev, &[1.0, 2.0]);
    let mut y = f32s(&dev, &[0.0, 0.0]);
    let mut double = scale_node(&dev, 2, 2.0);

    dev.scope(|s| -> Result<(), Error> {
        let (xs, ys) = (s.bind(&mut x)?, s.bind(&mut y)?);
        let double = s.bind_node(&mut double)?;
        s.run(double, &[Arg::Write(xs)])?;
        s.copy(xs, 0, ys, 0, 2)?;
        s.run(double, &[Arg::Write(xs)])?;
        assert!(s.copy(xs, 0, xs, 1, 1).is_err(), "same tensor");
        Ok(())
    })
    .unwrap();

    // y got x after the first run only
    assert_eq!(y.download_vec::<F32>(), Ok(vec![2.0, 4.0]));
    assert_eq!(x.download_vec::<F32>(), Ok(vec![4.0, 8.0]));
}

#[test]
fn node_meta_reads_back_as_built_and_updated() {
    let dev = device(0);
    let wg = Workgroups { x: 1, y: 1, z: 1 };
    let mut node = dev
        .node(
            &SCALE_LIKE,
            &[MetaValue::Uint(4), MetaValue::Float(2.0)],
            wg,
        )
        .unwrap();
    assert_eq!(node.meta(), &[MetaValue::Uint(4), MetaValue::Float(2.0)]);
    node.update_meta(&[MetaValue::Uint(3), MetaValue::Float(0.5)])
        .unwrap();
    assert_eq!(node.meta(), &[MetaValue::Uint(3), MetaValue::Float(0.5)]);
    assert!(node.update_meta(&[MetaValue::Uint(3)]).is_err());
    assert_eq!(
        node.meta()[0],
        MetaValue::Uint(3),
        "a failed update keeps the meta"
    );
}

#[test]
fn tensors_and_nodes_know_their_device() {
    let (a, b) = (device(0), device(0));
    let tensor = a.alloc(DataKind::F32, 1).unwrap();
    let node = b
        .node(
            &SCALE_LIKE,
            &[MetaValue::Uint(1), MetaValue::Float(1.0)],
            Workgroups { x: 1, y: 1, z: 1 },
        )
        .unwrap();
    assert!(std::ptr::eq(tensor.device(), &a));
    assert!(std::ptr::eq(node.device(), &b));
}

#[test]
fn tensor_replace_checks_device_and_kind() {
    let (a, b) = (device(0), device(0));
    let mut tensor = f32s(&a, &[1.0]);
    let old = tensor
        .replace(f32s(&a, &[2.0, 3.0]))
        .map_err(|(_, e)| e)
        .unwrap();
    assert_eq!(old.download_vec::<F32>().unwrap(), vec![1.0]);
    assert_eq!(tensor.download_vec::<F32>().unwrap(), vec![2.0, 3.0]);

    let (back, _) = tensor
        .replace(f32s(&b, &[4.0]))
        .err()
        .expect("replace fails");
    assert!(std::ptr::eq(back.device(), &b));
    let (back, _) = tensor
        .replace(a.alloc(DataKind::Int8, 1).unwrap())
        .err()
        .expect("replace fails");
    assert_eq!(back.kind(), DataKind::Int8);
    assert_eq!(tensor.download_vec::<F32>().unwrap(), vec![2.0, 3.0]);
}

static SCALE_UINT: Shader = Shader {
    name: "scale_uint",
    meta: &[
        MetaField {
            name: "n",
            ty: MetaType::Uint,
        },
        MetaField {
            name: "factor",
            ty: MetaType::Uint,
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

#[test]
fn node_replace_checks_device_slots_and_meta_types() {
    let (a, b) = (device(0), device(0));
    let wg = Workgroups { x: 1, y: 1, z: 1 };
    let mut node = scale_node(&a, 1, 2.0);
    let old = node
        .replace(scale_node(&a, 2, 3.0))
        .map_err(|(_, e)| e)
        .unwrap();
    assert_eq!(old.meta(), &[MetaValue::Uint(1), MetaValue::Float(2.0)]);
    assert_eq!(node.meta(), &[MetaValue::Uint(2), MetaValue::Float(3.0)]);

    assert!(node.replace(scale_node(&b, 1, 2.0)).is_err());
    let add = a.node(&ADD_LIKE, &[MetaValue::Uint(1)], wg).unwrap();
    assert!(node.replace(add).is_err());
    let uint = a
        .node(&SCALE_UINT, &[MetaValue::Uint(1), MetaValue::Uint(2)], wg)
        .unwrap();
    let (back, _) = node.replace(uint).err().expect("replace fails");
    assert_eq!(back.shader().name, "scale_uint");
    assert_eq!(node.meta(), &[MetaValue::Uint(2), MetaValue::Float(3.0)]);
}

#[test]
fn node_rejects_zero_workgroups() {
    let dev = device(0);
    let meta = [MetaValue::Uint(1), MetaValue::Float(1.0)];
    for wg in [
        Workgroups { x: 0, y: 1, z: 1 },
        Workgroups { x: 1, y: 0, z: 1 },
        Workgroups { x: 1, y: 1, z: 0 },
    ] {
        let err = dev.node(&SCALE_LIKE, &meta, wg).err();
        assert_eq!(
            err,
            Some(Error::NoWorkgroups {
                shader: "scale_like"
            })
        );
    }
}
