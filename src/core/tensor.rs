//! A tensor, blueprint and live: what it will be before any buffer exists,
//! and the handle to it once one does.

use crate::core::device::Device;
use crate::core::dtype::{DataKind, DataType, HostData};
use crate::core::id::GlobalId;
use crate::core::shader::ResolvedSize;

pub type TensorId = GlobalId<TensorSpec>;

#[derive(Clone)]
pub enum TensorSize {
    Fixed(u32),
    /// Not known at compile time
    Resolvable(ResolvedSize),
}

impl TensorSize {
    /// # Panics
    /// If it's `Resolvable` and hasn't been resolved yet.
    pub fn resolved(&self) -> u32 {
        match self {
            TensorSize::Fixed(n) => *n,
            TensorSize::Resolvable(r) => r.size.value() * r.multiplier + r.coefficient,
        }
    }
}

pub enum InitRecipe {
    UploadFromHost(HostData),
    Zero,
}

pub struct TensorSpec {
    id: TensorId,
    kind: DataKind,
    size: TensorSize,
    init: Option<InitRecipe>,
}

impl TensorSpec {
    pub fn blank<D: DataType>(size: TensorSize) -> Self {
        Self::with::<D>(size, None)
    }

    pub fn seeded<D: DataType>(size: TensorSize, data: Vec<D::HostRepr>) -> Self {
        Self::with::<D>(size, Some(InitRecipe::UploadFromHost(D::wrap(data))))
    }

    pub fn zeroed<D: DataType>(size: TensorSize) -> Self {
        Self::with::<D>(size, Some(InitRecipe::Zero))
    }

    fn with<D: DataType>(size: TensorSize, init: Option<InitRecipe>) -> Self {
        Self {
            id: GlobalId::new(),
            kind: D::KIND,
            size,
            init,
        }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn kind(&self) -> DataKind {
        self.kind
    }

    pub fn size(&self) -> &TensorSize {
        &self.size
    }

    pub fn init(&self) -> Option<&InitRecipe> {
        self.init.as_ref()
    }
}

/// A live tensor on one device: just a handle (device + id)
/// the buffer itself lives in the device's own table.
///
/// Not `Clone`
pub struct Tensor {
    device: Device,
    id: TensorId,
    kind: DataKind,
    elem_count: usize,
}

impl Tensor {
    pub fn new(device: &Device, kind: DataKind, elem_count: usize) -> Result<Self, String> {
        Self::with_id(device, TensorId::new(), kind, elem_count)
    }

    /// Like `new`, with the id fixed instead of freshly minted.
    /// For a `SpreadSpec`'s parts and whole, whose ids are decided up front.
    pub(crate) fn with_id(
        device: &Device,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<Self, String> {
        device.alloc_kind(id, kind, elem_count)?;
        Ok(Self {
            device: device.clone(),
            id,
            kind,
            elem_count,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn kind(&self) -> DataKind {
        self.kind
    }

    pub fn len(&self) -> usize {
        self.elem_count
    }

    // Panics instead of an Err on a wrong `D`
    fn assert_kind<D: DataType>(&self) {
        assert_eq!(
            D::KIND,
            self.kind,
            "tensor holds {:?}, accessed as {:?}",
            self.kind,
            D::KIND
        );
    }

    /// Host words this tensor spans
    ///
    /// differs from `len` only for packed
    /// kinds (Int4: 8 values per word).
    fn host_len<D: DataType>(&self) -> usize {
        self.elem_count.div_ceil(D::ELEMS_PER_HOST_WORD as usize)
    }

    /// # Panics
    /// If `D` isn't this tensor's kind or `data` isn't exactly its size.
    pub fn upload<D: DataType>(&self, data: &[D::HostRepr]) {
        self.assert_kind::<D>();
        assert_eq!(
            data.len(),
            self.host_len::<D>(),
            "upload of {} host words into a tensor of {}",
            data.len(),
            self.host_len::<D>()
        );
        self.device
            .upload_kind(self.id, &D::wrap(data.to_vec()))
            .expect("the kind was accepted when this tensor was allocated");
    }

    /// # Panics
    /// If `D` isn't this tensor's kind.
    pub fn download<D: DataType>(&self) -> Vec<D::HostRepr> {
        self.assert_kind::<D>();
        let data = self
            .device
            .download_kind(self.id, self.kind)
            .expect("the kind was accepted when this tensor was allocated");
        D::unwrap(data).expect("the device returned the kind it was asked for")
    }

    /// A full copy on `to`, as a new tensor.
    /// Goes through the host for now;
    /// a same-backend device-to-device path comes with integration.
    pub fn copy_to<D: DataType>(&self, to: &Device) -> Result<Tensor, String> {
        self.copy_to_id::<D>(to, TensorId::new())
    }

    /// Like `copy_to`, with the copy's id fixed instead of freshly minted.
    pub(crate) fn copy_to_id<D: DataType>(
        &self,
        to: &Device,
        id: TensorId,
    ) -> Result<Tensor, String> {
        let copy = Tensor::with_id(to, id, self.kind, self.elem_count)?;
        copy.upload::<D>(&self.download::<D>());
        Ok(copy)
    }

    /// Writes this tensor into `dst`, starting at element `offset`. Host
    /// round trip for now, like `copy_to`.
    pub fn copy_into<D: DataType>(&self, dst: &Tensor, offset: usize) -> Result<(), String> {
        let per_word = D::ELEMS_PER_HOST_WORD as usize;
        if offset % per_word != 0 {
            return Err(format!(
                "offset {offset} isn't a multiple of {per_word}, the packing of {:?}",
                D::KIND
            ));
        }
        if offset + self.elem_count > dst.elem_count {
            return Err(format!(
                "copying {} elements at offset {offset} overruns a tensor of {}",
                self.elem_count, dst.elem_count
            ));
        }
        let mut whole = dst.download::<D>();
        let start = offset / per_word;
        let part = self.download::<D>();
        whole[start..start + part.len()].copy_from_slice(&part);
        dst.upload::<D>(&whole);
        Ok(())
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        self.device.drop_buffer(self.id);
    }
}

#[cfg(test)]
#[path = "../tests/tensor.rs"]
mod tests;
