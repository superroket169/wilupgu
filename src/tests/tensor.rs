use super::*;
use crate::backend::dtype::{DataType, Int4, F32};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::DeviceConfig;

fn device() -> ComputeDevice<CpuBackend> {
    ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    )
}

fn spec(kind: DataKind, len: Resolvable<usize>, init: TensorInit) -> Result<TensorSpec, String> {
    TensorSpec::new(TensorDesc { kind, len, init })
}

#[test]
fn new_rejects_data_of_another_kind() {
    let data = Int4::wrap(vec![0]);
    assert!(spec(
        DataKind::F32,
        Resolvable::fixed(8),
        TensorInit::Upload(data)
    )
    .is_err());
}

#[test]
fn ids_are_unique() {
    let a = spec(DataKind::F32, Resolvable::fixed(1), TensorInit::Blank).unwrap();
    let b = spec(DataKind::F32, Resolvable::fixed(1), TensorInit::Blank).unwrap();
    assert_ne!(a.id(), b.id());
}

#[test]
fn ready_waits_for_the_length() {
    let len = Resolvable::new();
    let resolver = len.resolver().unwrap();
    let s = spec(DataKind::F32, len, TensorInit::Zero).unwrap();
    assert!(s.ready().is_err());
    assert!(s.build(&device()).is_err());
    resolver.resolve(4);
    assert!(s.ready().is_ok());
}

#[test]
fn ready_checks_the_upload_length_once_known() {
    let s = spec(
        DataKind::F32,
        Resolvable::fixed(3),
        TensorInit::Upload(F32::wrap(vec![1.0, 2.0])),
    )
    .unwrap();
    assert!(s.ready().is_err());
}

#[test]
fn packed_upload_length_is_rounded_to_whole_words() {
    let s = spec(
        DataKind::Int4,
        Resolvable::fixed(12),
        TensorInit::Upload(Int4::wrap(vec![0x7654_3210, 0xfedc_ba98])),
    )
    .unwrap();
    let dev = device();
    let t = s.build(&dev).unwrap();
    assert_eq!(t.download_vec::<Int4>(), Ok(vec![0x7654_3210, 0xfedc_ba98]));
}

#[test]
fn build_follows_the_init() {
    let dev = device();
    let uploaded = spec(
        DataKind::F32,
        Resolvable::fixed(3),
        TensorInit::Upload(F32::wrap(vec![1.0, 2.0, 3.0])),
    )
    .unwrap()
    .build(&dev)
    .unwrap();
    assert_eq!(uploaded.download_vec::<F32>(), Ok(vec![1.0, 2.0, 3.0]));

    // a pooled buffer comes back dirty; Zero must still clear it
    let dev = ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig {
            pool_bytes: 1 << 20,
        },
    );
    drop(dev.tensor_from(&F32::wrap(vec![5.0; 4])).unwrap());
    let zeroed = spec(DataKind::F32, Resolvable::fixed(4), TensorInit::Zero)
        .unwrap()
        .build(&dev)
        .unwrap();
    assert_eq!(zeroed.download_vec::<F32>(), Ok(vec![0.0; 4]));

    let blank = spec(DataKind::F32, Resolvable::fixed(5), TensorInit::Blank)
        .unwrap()
        .build(&dev)
        .unwrap();
    assert_eq!((blank.kind(), blank.len()), (DataKind::F32, 5));
}

#[test]
fn specs_sharing_a_length_see_one_value() {
    let batch = Resolvable::new();
    let a = spec(DataKind::F32, batch.clone(), TensorInit::Blank).unwrap();
    let b = spec(DataKind::F16, batch.clone(), TensorInit::Blank).unwrap();
    batch.resolver().unwrap().resolve(256);
    assert_eq!(a.len().value(), Ok(&256));
    assert_eq!(b.len().value(), Ok(&256));
}
