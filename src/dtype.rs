#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    F32,
    F16,
    Bf16,
    Int8,
    Int4,
}

pub trait DataType: Copy + Send + Sync + 'static {
    type HostRepr: bytemuck::Pod + Default + Clone;
    const BITS_PER_ELEM: u32;
    const ELEMS_PER_HOST_WORD: u32 = 1;
    const KIND: DataKind;

    fn wrap(data: Vec<Self::HostRepr>) -> HostData;
    /// `None` if `data` holds a different kind.
    fn unwrap(data: HostData) -> Option<Vec<Self::HostRepr>>;
}

#[derive(Clone, Copy)]
pub struct F32;
impl DataType for F32 {
    type HostRepr = f32;
    const BITS_PER_ELEM: u32 = 32;
    const KIND: DataKind = DataKind::F32;

    fn wrap(data: Vec<f32>) -> HostData {
        HostData::F32(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<f32>> {
        match data {
            HostData::F32(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct F16;
impl DataType for F16 {
    type HostRepr = half::f16;
    const BITS_PER_ELEM: u32 = 16;
    const KIND: DataKind = DataKind::F16;

    fn wrap(data: Vec<half::f16>) -> HostData {
        HostData::F16(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<half::f16>> {
        match data {
            HostData::F16(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Bf16;
impl DataType for Bf16 {
    type HostRepr = half::bf16;
    const BITS_PER_ELEM: u32 = 16;
    const KIND: DataKind = DataKind::Bf16;

    fn wrap(data: Vec<half::bf16>) -> HostData {
        HostData::Bf16(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<half::bf16>> {
        match data {
            HostData::Bf16(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Int8;
impl DataType for Int8 {
    type HostRepr = u8;
    const BITS_PER_ELEM: u32 = 8;
    const KIND: DataKind = DataKind::Int8;

    fn wrap(data: Vec<u8>) -> HostData {
        HostData::Int8(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<u8>> {
        match data {
            HostData::Int8(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Int4;
impl DataType for Int4 {
    type HostRepr = u32;
    const BITS_PER_ELEM: u32 = 4;
    const ELEMS_PER_HOST_WORD: u32 = 8;
    const KIND: DataKind = DataKind::Int4;

    fn wrap(data: Vec<u32>) -> HostData {
        HostData::Int4(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<u32>> {
        match data {
            HostData::Int4(v) => Some(v),
            _ => None,
        }
    }
}

/// Host-side values of one tensor
/// one tag for the whole vector
/// so kinds can't mix and upload is a plain byte cast.
#[derive(Debug, Clone)]
pub enum HostData {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    Int8(Vec<u8>),
    /// Packed, 8 values per word.
    Int4(Vec<u32>),
}

impl HostData {
    pub fn kind(&self) -> DataKind {
        match self {
            HostData::F32(_) => DataKind::F32,
            HostData::F16(_) => DataKind::F16,
            HostData::Bf16(_) => DataKind::Bf16,
            HostData::Int8(_) => DataKind::Int8,
            HostData::Int4(_) => DataKind::Int4,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
