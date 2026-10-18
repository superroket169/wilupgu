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

#[test]
fn bytes_for_follows_bits_per_elem() {
    assert_eq!(DataKind::F32.bytes_for(0), 0);
    assert_eq!(DataKind::F32.bytes_for(3), 12);
    assert_eq!(DataKind::F16.bytes_for(3), 6);
    assert_eq!(DataKind::Int8.bytes_for(5), 5);
    assert_eq!(DataKind::Int4.bytes_for(16), 8);
}

#[test]
fn host_data_len_counts_elements_not_words() {
    assert_eq!(F32::wrap(vec![1.0, 2.0]).len(), 2);
    assert_eq!(Int4::wrap(vec![0, 0]).len(), 16);
    assert!(Int8::wrap(vec![]).is_empty());
}

#[test]
fn from_bytes_is_the_inverse_of_as_bytes() {
    let cases = [
        F32::wrap(vec![1.0, -2.5]),
        F16::wrap(vec![half::f16::from_f32(0.5)]),
        Bf16::wrap(vec![half::bf16::from_f32(3.0)]),
        Int8::wrap(vec![1, 2, 3]),
        Int4::wrap(vec![0x7654_3210]),
    ];
    for data in cases {
        let back = HostData::from_bytes(data.kind(), data.as_bytes().to_vec());
        assert_eq!(back.kind(), data.kind());
        assert_eq!(back.as_bytes(), data.as_bytes());
    }
}
