//! The data types a buffer can hold: as a runtime value (`DataKind`), as a
//! type (`DataType` and its markers) and as host-side values (`HostData`).

/// A data type as a runtime value, for code that only learns the type at
/// runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataKind {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// 16-bit bfloat.
    Bf16,
    /// 8-bit integer.
    Int8,
    /// 4-bit integer, packed 8 to a 32-bit word.
    Int4,
}

/// A data type as a type: one zero-sized marker per kind (`F32`, `F16`, ...),
/// for code that knows the type at compile time.
pub trait DataType: Copy + Send + Sync + 'static {
    /// One host word of this type.
    type HostRepr: bytemuck::Pod + Default + Clone;
    /// Bits one element takes.
    const BITS_PER_ELEM: u32;
    /// Elements in one `HostRepr` word; more than 1 only for packed kinds.
    const ELEMS_PER_HOST_WORD: u32 = 1;
    /// This type's `DataKind`.
    const KIND: DataKind;

    /// Tags host words as `HostData` of this kind.
    fn wrap(data: Vec<Self::HostRepr>) -> HostData;
    /// `None` if `data` holds a different kind.
    fn unwrap(data: HostData) -> Option<Vec<Self::HostRepr>>;
}

/// `f32` elements.
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

/// 16-bit IEEE float elements, `half::f16` on the host.
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

/// 16-bit bfloat elements, `half::bf16` on the host.
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

/// 8-bit integer elements, `u8` on the host.
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

/// 4-bit integer elements, packed 8 to a `u32` word on the host.
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
    /// `F32` values.
    F32(Vec<f32>),
    /// `F16` values.
    F16(Vec<half::f16>),
    /// `Bf16` values.
    Bf16(Vec<half::bf16>),
    /// `Int8` values.
    Int8(Vec<u8>),
    /// Packed, 8 values per word.
    Int4(Vec<u32>),
}

impl HostData {
    /// The `DataKind` of these values.
    pub fn kind(&self) -> DataKind {
        match self {
            HostData::F32(_) => DataKind::F32,
            HostData::F16(_) => DataKind::F16,
            HostData::Bf16(_) => DataKind::Bf16,
            HostData::Int8(_) => DataKind::Int8,
            HostData::Int4(_) => DataKind::Int4,
        }
    }

    pub fn len(&self) -> usize {
        let words = match self {
            HostData::F32(v) => v.len(),
            HostData::F16(v) => v.len(),
            HostData::Bf16(v) => v.len(),
            HostData::Int8(v) => v.len(),
            HostData::Int4(v) => v.len(),
        };
        words * self.kind().elems_per_host_word()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            HostData::F32(v) => bytemuck::cast_slice(v),
            HostData::F16(v) => bytemuck::cast_slice(v),
            HostData::Bf16(v) => bytemuck::cast_slice(v),
            HostData::Int8(v) => v,
            HostData::Int4(v) => bytemuck::cast_slice(v),
        }
    }

    // Copies into a fresh `Vec` of the right type, so `bytes` needs no alignment.
    pub fn from_bytes(kind: DataKind, bytes: Vec<u8>) -> HostData {
        match kind {
            DataKind::F32 => F32::wrap(bytemuck::pod_collect_to_vec(&bytes)),
            DataKind::F16 => F16::wrap(bytemuck::pod_collect_to_vec(&bytes)),
            DataKind::Bf16 => Bf16::wrap(bytemuck::pod_collect_to_vec(&bytes)),
            DataKind::Int8 => Int8::wrap(bytes),
            DataKind::Int4 => Int4::wrap(bytemuck::pod_collect_to_vec(&bytes)),
        }
    }
}

impl DataKind {
    pub fn bits_per_elem(self) -> u32 {
        match self {
            DataKind::F32 => F32::BITS_PER_ELEM,
            DataKind::F16 => F16::BITS_PER_ELEM,
            DataKind::Bf16 => Bf16::BITS_PER_ELEM,
            DataKind::Int8 => Int8::BITS_PER_ELEM,
            DataKind::Int4 => Int4::BITS_PER_ELEM,
        }
    }

    pub fn elems_per_host_word(self) -> usize {
        let n = match self {
            DataKind::F32 => F32::ELEMS_PER_HOST_WORD,
            DataKind::F16 => F16::ELEMS_PER_HOST_WORD,
            DataKind::Bf16 => Bf16::ELEMS_PER_HOST_WORD,
            DataKind::Int8 => Int8::ELEMS_PER_HOST_WORD,
            DataKind::Int4 => Int4::ELEMS_PER_HOST_WORD,
        };
        n as usize
    }

    pub fn bytes_for(self, len: usize) -> u64 {
        (len as u64 * self.bits_per_elem() as u64).div_ceil(8)
    }
}

#[cfg(test)]
#[path = "../tests/dtype.rs"]
mod tests;
