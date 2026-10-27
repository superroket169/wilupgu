use super::*;
use crate::backend::dtype::DataKind;
use crate::backend::shader::{MetaField, MetaType, Native, NativeBinding, ShaderBinary};
use crate::backends::CpuBackend;

fn each_n(m: &[u32], _: Workgroups, l: &mut [usize]) {
    l.fill(m[0] as usize);
}

// The input holds `n`, the output two partials.
fn two_partials(m: &[u32], _: Workgroups, l: &mut [usize]) {
    l.copy_from_slice(&[m[0] as usize, 2]);
}

const N: [MetaField; 1] = [MetaField {
    name: "n",
    ty: MetaType::Uint,
}];

static ADD_LIKE: Signature = Signature {
    name: "add_like",
    meta: &N,
    layout: &[
        BindingRole::Accumulate(DataKind::F32),
        BindingRole::Input(DataKind::F32),
    ],
    min_lens: each_n,
    workgroup_size: [256, 1, 1],
};

static FILL_LIKE: Signature = Signature {
    name: "fill_like",
    meta: &N,
    layout: &[BindingRole::Output(DataKind::F32)],
    min_lens: each_n,
    workgroup_size: [256, 1, 1],
};

static SUM_LIKE: Signature = Signature {
    name: "sum_like",
    meta: &N,
    layout: &[
        BindingRole::Input(DataKind::F32),
        BindingRole::Output(DataKind::F32),
    ],
    min_lens: two_partials,
    workgroup_size: [256, 1, 1],
};

static F16_LIKE: Signature = Signature {
    name: "f16_like",
    meta: &N,
    layout: &[BindingRole::Output(DataKind::F16)],
    min_lens: each_n,
    workgroup_size: [256, 1, 1],
};

fn written<'a>(b: &'a mut NativeBinding) -> &'a mut [f32] {
    match b {
        NativeBinding::Write(b) => bytemuck::cast_slice_mut(b),
        NativeBinding::Read(_) => panic!("a read slot"),
    }
}

fn read(b: &NativeBinding) -> Vec<f32> {
    let bytes: &[u8] = match b {
        NativeBinding::Read(b) => b,
        NativeBinding::Write(b) => b,
    };
    bytemuck::pod_collect_to_vec(bytes)
}

fn add(meta: &[u32], b: &mut [NativeBinding]) {
    let y = read(&b[1]);
    for (x, y) in written(&mut b[0]).iter_mut().zip(y).take(meta[0] as usize) {
        *x += y;
    }
}

fn add_and_a_bit(meta: &[u32], b: &mut [NativeBinding]) {
    add(meta, b);
    written(&mut b[0])[0] += 1e-3;
}

fn fill(meta: &[u32], b: &mut [NativeBinding]) {
    written(&mut b[0])[..meta[0] as usize].fill(1.0);
}

fn fill_by_adding(meta: &[u32], b: &mut [NativeBinding]) {
    for x in &mut written(&mut b[0])[..meta[0] as usize] {
        *x += 1.0;
    }
}

fn fill_half(meta: &[u32], b: &mut [NativeBinding]) {
    written(&mut b[0])[..meta[0] as usize / 2].fill(1.0);
}

fn fill_one_more(meta: &[u32], b: &mut [NativeBinding]) {
    written(&mut b[0])[..meta[0] as usize + 1].fill(1.0);
}

fn sum_in_one(meta: &[u32], b: &mut [NativeBinding]) {
    let x = read(&b[0]);
    written(&mut b[1])[0] = x[..meta[0] as usize].iter().sum();
}

fn sum_in_two(meta: &[u32], b: &mut [NativeBinding]) {
    let x = read(&b[0]);
    let (lo, hi) = x[..meta[0] as usize].split_at(meta[0] as usize / 2);
    let out = written(&mut b[1]);
    out[0] = lo.iter().sum();
    out[1] = hi.iter().sum();
}

fn shader(signature: &'static Signature, f: fn(&[u32], &mut [NativeBinding])) -> Shader {
    Shader::new(signature, ShaderBinary::Native(Native(f)))
}

fn devices() -> [ComputeDevice<CpuBackend>; 2] {
    [(); 2].map(|_| ComputeDevice::first().unwrap())
}

fn n(n: u32) -> [MetaValue; 1] {
    [MetaValue::Uint(n)]
}

const EXACT: Check = Check::Close {
    atol: 0.0,
    rtol: 0.0,
};

#[test]
fn the_same_shader_on_two_devices_passes() {
    let [a, b] = devices();
    let add = shader(&ADD_LIKE, add);
    let case = Case::new(&n(300), Workgroups::linear(2), &[EXACT, Check::Ignore]).with_seed(7);
    let report = compare((&a, &add), (&b, &add), &case).unwrap();
    assert!(report.passed(), "{report}");
    assert_eq!(report.backends(), ["cpu", "cpu"]);
    assert_eq!(report.slots()[0].diff().unwrap().err(), 0.0);
    assert_eq!(report.slots()[0].len(), 300);
    assert!(report.slots()[1].diff().is_none());
}

#[test]
fn a_difference_fails_past_its_tolerance_only() {
    let [a, b] = devices();
    let (add, off) = (shader(&ADD_LIKE, add), shader(&ADD_LIKE, add_and_a_bit));
    let within = |atol| {
        let checks = [Check::Close { atol, rtol: 0.0 }, Check::Ignore];
        compare(
            (&a, &add),
            (&b, &off),
            &Case::new(&n(10), Workgroups::linear(1), &checks),
        )
        .unwrap()
    };
    let tight = within(1e-4);
    assert!(!tight.passed());
    assert_eq!(tight.slots()[0].diff().unwrap().index(), Some(0));
    assert!(tight.to_string().contains("FAIL"), "{tight}");
    assert!(within(1e-2).passed());
}

#[test]
fn the_canary_catches_an_output_that_is_read() {
    let [a, b] = devices();
    let (ok, reads) = (shader(&FILL_LIKE, fill), shader(&FILL_LIKE, fill_by_adding));
    let case = Case::new(&n(8), Workgroups::linear(1), &[Check::Ignore]);
    let report = compare((&a, &ok), (&b, &reads), &case).unwrap();
    assert_eq!(report.slots()[0].unwritten(), [None, Some(0)]);
    assert!(!report.passed());
}

#[test]
fn the_canary_catches_an_output_left_half_written() {
    let [a, b] = devices();
    let (ok, half) = (shader(&FILL_LIKE, fill), shader(&FILL_LIKE, fill_half));
    let case = Case::new(&n(8), Workgroups::linear(1), &[Check::Ignore]);
    let report = compare((&a, &half), (&b, &ok), &case).unwrap();
    assert_eq!(report.slots()[0].unwritten(), [Some(4), None]);
}

#[test]
fn the_pad_catches_a_write_past_min_lens() {
    let [a, b] = devices();
    let (ok, over) = (shader(&FILL_LIKE, fill), shader(&FILL_LIKE, fill_one_more));
    let case = Case::new(&n(8), Workgroups::linear(1), &[EXACT]);
    let report = compare((&a, &ok), (&b, &over), &case).unwrap();
    assert_eq!(report.slots()[0].overrun(), [None, Some(8)]);
    assert!(report.slots()[0].diff().unwrap().passed());
    assert!(!report.passed());
    assert!(report
        .to_string()
        .contains("cpu wrote past min_lens at [8]"));
}

#[test]
fn total_compares_sums_not_elements() {
    let [a, b] = devices();
    let (one, two) = (shader(&SUM_LIKE, sum_in_one), shader(&SUM_LIKE, sum_in_two));
    let run = |check| {
        let case = Case::new(&n(100), Workgroups::linear(1), &[Check::Ignore, check]);
        compare((&a, &one), (&b, &two), &case).unwrap()
    };
    assert!(!run(EXACT).passed());
    let total = run(Check::Total {
        atol: 1e-4,
        rtol: 0.0,
    });
    assert!(total.passed(), "{total}");
    assert_eq!(total.slots()[1].diff().unwrap().index(), None);
}

#[test]
fn an_empty_slot_has_nothing_to_compare() {
    let [a, b] = devices();
    let fill = shader(&FILL_LIKE, fill);
    let case = Case::new(&n(0), Workgroups::linear(1), &[EXACT]);
    let report = compare((&a, &fill), (&b, &fill), &case).unwrap();
    assert!(report.slots()[0].is_empty());
    assert!(report.slots()[0].diff().is_none());
    assert!(report.passed());
}

#[test]
fn seeded_data_repeats_and_stays_in_range() {
    assert_eq!(random(3, 1, 100), random(3, 1, 100));
    assert_ne!(random(3, 1, 100), random(4, 1, 100));
    assert_ne!(random(3, 1, 100), random(3, 2, 100));
    assert!(random(3, 1, 10_000).iter().all(|v| (-1.0..1.0).contains(v)));
}

#[test]
fn other_signatures_fail() {
    let [a, b] = devices();
    let case = Case::new(&n(1), Workgroups::linear(1), &[EXACT]);
    let err = compare(
        (&a, &shader(&FILL_LIKE, fill)),
        (&b, &shader(&ADD_LIKE, add)),
        &case,
    );
    assert_eq!(
        err,
        Err(Error::SignatureMismatch {
            a: "fill_like",
            b: "add_like"
        })
    );
}

#[test]
fn a_check_per_slot_or_it_fails() {
    let [a, b] = devices();
    let add = shader(&ADD_LIKE, add);
    let case = Case::new(&n(1), Workgroups::linear(1), &[EXACT]);
    assert_eq!(
        compare((&a, &add), (&b, &add), &case),
        Err(Error::CheckCount {
            shader: "add_like",
            expected: 2,
            got: 1
        })
    );
}

#[test]
fn a_slot_of_another_kind_than_f32_fails() {
    let [a, b] = devices();
    let f16 = shader(&F16_LIKE, fill);
    let case = Case::new(&n(1), Workgroups::linear(1), &[Check::Ignore]);
    assert_eq!(
        compare((&a, &f16), (&b, &f16), &case),
        Err(Error::ParityKind {
            shader: "f16_like",
            slot: 1,
            kind: DataKind::F16
        })
    );
}
