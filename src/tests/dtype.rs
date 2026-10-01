use super::*;

#[test]
fn host_data_kind_follows_its_variant() {
    assert_eq!(F16::wrap(vec![]).kind(), DataKind::F16);
    assert_eq!(Int4::wrap(vec![0]).kind(), DataKind::Int4);
}

#[test]
fn unwrap_is_the_inverse_of_wrap() {
    assert_eq!(F32::unwrap(F32::wrap(vec![1.0, 2.0])), Some(vec![1.0, 2.0]));
    assert_eq!(F32::unwrap(F16::wrap(vec![])), None);
}
