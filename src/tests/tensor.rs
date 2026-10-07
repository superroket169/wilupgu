use super::*;
use crate::core::deferred::Resolvable;
use crate::core::dtype::{Int8, F16, F32};

#[test]
fn specs_sharing_a_size_see_one_value() {
    let batch = Resolvable::new();
    let a = TensorSpec::blank::<F32>(batch.clone());
    let b = TensorSpec::blank::<F16>(batch.clone());
    batch.resolver().resolve(256);
    assert_eq!(*a.size().value(), 256);
    assert_eq!(*b.size().value(), 256);
}

#[test]
fn tensor_spec_blank_has_no_init() {
    let spec = TensorSpec::blank::<F32>(Resolvable::fixed(4));
    assert!(spec.init().is_none());
    assert_eq!(spec.kind(), DataKind::F32);
}

#[test]
fn tensor_spec_seeded_carries_its_init() {
    let spec = TensorSpec::seeded::<F32>(Resolvable::fixed(4), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        spec.init(),
        Some(InitRecipe::UploadFromHost(HostData::F32(data))) if data.len() == 4
    ));
}

#[test]
fn tensor_spec_zeroed_keeps_its_kind() {
    let spec = TensorSpec::zeroed::<F16>(Resolvable::fixed(4));
    assert!(matches!(spec.init(), Some(InitRecipe::Zero)));
    assert_eq!(spec.kind(), DataKind::F16);
}

#[test]
fn specs_of_different_kinds_share_one_slice() {
    let specs = [
        TensorSpec::blank::<F32>(Resolvable::fixed(4)),
        TensorSpec::seeded::<Int8>(Resolvable::fixed(2), vec![1, 2]),
    ];
    assert_eq!(specs[0].kind(), DataKind::F32);
    assert_eq!(specs[1].kind(), DataKind::Int8);
}

#[test]
fn tensor_spec_ids_are_distinct() {
    let a = TensorSpec::blank::<F32>(Resolvable::fixed(4));
    let b = TensorSpec::blank::<F32>(Resolvable::fixed(4));
    assert_ne!(a.id(), b.id());
}

use crate::backend::Storage;
use crate::backends::toy::ToyBackend;

fn toy(device: &Device) -> &ToyBackend {
    match device {
        Device::Toy(t) => t,
        #[allow(unreachable_patterns)]
        _ => unreachable!(),
    }
}

#[test]
fn upload_then_download_round_trips() {
    let device = Device::Toy(ToyBackend::new());
    let mut t = Tensor::new(&device, DataKind::F32, 3).unwrap();
    t.upload::<F32>(&[1.0, 2.0, 3.0]);
    assert_eq!(t.download::<F32>(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn new_rejects_a_kind_the_backend_lacks() {
    let device = Device::Toy(ToyBackend::new());
    assert!(Tensor::new(&device, DataKind::F16, 1).is_err());
}

#[test]
fn drop_sends_the_buffer_to_the_pool() {
    let device = Device::Toy(ToyBackend::new());
    let t = Tensor::new(&device, DataKind::F32, 4).unwrap();
    drop(t);
    assert!(toy(&device).table().take_free(DataKind::F32, 4).is_some());
}

#[test]
fn free_succeeds_when_nothing_holds_the_buffer() {
    let device = Device::Toy(ToyBackend::new());
    let t = Tensor::new(&device, DataKind::F32, 4).unwrap();
    t.free().unwrap();
    assert!(toy(&device).table().take_free(DataKind::F32, 4).is_some());
}

#[test]
fn free_fails_while_a_node_holds_the_buffer() {
    let device = Device::Toy(ToyBackend::new());
    let t = Tensor::new(&device, DataKind::F32, 4).unwrap();
    let held = toy(&device).table().get(t.id).unwrap(); // stands in for a built node's clone
    let err = t.free().unwrap_err();
    assert!(err.contains("still in use"), "{err}");
    assert!(toy(&device).table().take_free(DataKind::F32, 4).is_none());
    drop(held);
}

#[test]
fn copy_to_moves_the_data_to_another_device() {
    let (a, b) = (
        Device::Toy(ToyBackend::new()),
        Device::Toy(ToyBackend::new()),
    );
    let mut t = Tensor::new(&a, DataKind::F32, 2).unwrap();
    t.upload::<F32>(&[5.0, 6.0]);
    let copy = t.copy_to::<F32>(&b).unwrap();
    assert_eq!(copy.device().id(), b.id());
    assert_eq!(copy.download::<F32>(), vec![5.0, 6.0]);
}

#[test]
fn copy_into_writes_at_the_offset() {
    let device = Device::Toy(ToyBackend::new());
    let mut part = Tensor::new(&device, DataKind::F32, 2).unwrap();
    part.upload::<F32>(&[7.0, 8.0]);
    let mut whole = Tensor::new(&device, DataKind::F32, 4).unwrap();
    whole.upload::<F32>(&[0.0; 4]);
    part.copy_into::<F32>(&mut whole, 1).unwrap();
    assert_eq!(whole.download::<F32>(), vec![0.0, 7.0, 8.0, 0.0]);
}

#[test]
fn copy_into_rejects_an_overrun() {
    let device = Device::Toy(ToyBackend::new());
    let part = Tensor::new(&device, DataKind::F32, 2).unwrap();
    let mut whole = Tensor::new(&device, DataKind::F32, 2).unwrap();
    assert!(part.copy_into::<F32>(&mut whole, 1).is_err());
}
