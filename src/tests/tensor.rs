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
