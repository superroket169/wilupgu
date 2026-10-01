use super::*;
use crate::core::dtype::{Int8, F16, F32};
use crate::core::resolver::Resolvable;

#[test]
fn tensor_size_variants_construct() {
    let fixed = TensorSize::Fixed(128);
    let resolvable = TensorSize::Resolvable(ResolvedSize {
        size: Resolvable::new(),
        multiplier: 64,
        coefficient: 1,
    });
    assert!(matches!(fixed, TensorSize::Fixed(128)));
    assert!(matches!(resolvable, TensorSize::Resolvable(_)));
}

#[test]
fn tensor_size_resolvable_groups_by_shared_clone() {
    let size = Resolvable::new();
    let a = TensorSize::Resolvable(ResolvedSize {
        size: size.clone(),
        multiplier: 64,
        coefficient: 1,
    });
    let b = TensorSize::Resolvable(ResolvedSize {
        size: size.clone(),
        multiplier: 64,
        coefficient: 1,
    });
    size.resolver().resolve(256);
    let TensorSize::Resolvable(a_size) = a else {
        unreachable!()
    };
    let TensorSize::Resolvable(b_size) = b else {
        unreachable!()
    };
    assert_eq!(*a_size.size.value(), 256);
    assert_eq!(*b_size.size.value(), 256);
}

#[test]
fn tensor_spec_blank_has_no_init() {
    let spec = TensorSpec::blank::<F32>(TensorSize::Fixed(4));
    assert!(spec.init().is_none());
    assert_eq!(spec.kind(), DataKind::F32);
}

#[test]
fn tensor_spec_seeded_carries_its_init() {
    let spec = TensorSpec::seeded::<F32>(TensorSize::Fixed(4), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        spec.init(),
        Some(InitRecipe::UploadFromHost(HostData::F32(data))) if data.len() == 4
    ));
}

#[test]
fn tensor_spec_zeroed_keeps_its_kind() {
    let spec = TensorSpec::zeroed::<F16>(TensorSize::Fixed(4));
    assert!(matches!(spec.init(), Some(InitRecipe::Zero)));
    assert_eq!(spec.kind(), DataKind::F16);
}

#[test]
fn specs_of_different_kinds_share_one_slice() {
    let specs = [
        TensorSpec::blank::<F32>(TensorSize::Fixed(4)),
        TensorSpec::seeded::<Int8>(TensorSize::Fixed(2), vec![1, 2]),
    ];
    assert_eq!(specs[0].kind(), DataKind::F32);
    assert_eq!(specs[1].kind(), DataKind::Int8);
}

#[test]
fn tensor_spec_ids_are_distinct() {
    let a = TensorSpec::blank::<F32>(TensorSize::Fixed(4));
    let b = TensorSpec::blank::<F32>(TensorSize::Fixed(4));
    assert_ne!(a.id(), b.id());
}
