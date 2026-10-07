//! A tensor, blueprint and live: what it will be before any buffer exists,
//! and the handle to it once one does.

use crate::backend::dtype::{DataKind, DataType, HostData};
use crate::backend::id::{GlobalId, TensorId};
use crate::tools::core::deferred::Resolvable;
use crate::tools::core::device::Device;

pub enum InitRecipe {
    UploadFromHost(HostData),
    Zero,
}

pub struct TensorSpec {
    id: TensorId,
    kind: DataKind,
    size: Resolvable<u32>,
    init: Option<InitRecipe>,
}

impl TensorSpec {
    pub fn blank<D: DataType>(size: Resolvable<u32>) -> Self {
        Self::with::<D>(size, None)
    }

    pub fn seeded<D: DataType>(size: Resolvable<u32>, data: Vec<D::HostRepr>) -> Self {
        Self::with::<D>(size, Some(InitRecipe::UploadFromHost(D::wrap(data))))
    }

    pub fn zeroed<D: DataType>(size: Resolvable<u32>) -> Self {
        Self::with::<D>(size, Some(InitRecipe::Zero))
    }

    fn with<D: DataType>(size: Resolvable<u32>, init: Option<InitRecipe>) -> Self {
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

    pub fn size(&self) -> &Resolvable<u32> {
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
pub struct Tensor<'d> {
    device: &'d Device,
    id: TensorId,
    kind: DataKind,
    elem_count: usize,
}

impl<'d> Tensor<'d> {
    pub fn new(device: &'d Device, kind: DataKind, elem_count: usize) -> Result<Self, String> {
        Self::with_id(device, TensorId::new(), kind, elem_count)
    }

    /// Like `new`, with the id fixed instead of freshly minted.
    /// For a `SpreadSpec`'s parts and whole, whose ids are decided up front.
    pub(crate) fn with_id(
        device: &'d Device,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<Self, String> {
        device.alloc_kind(id, kind, elem_count)?;
        Ok(Self {
            device,
            id,
            kind,
            elem_count,
        })
    }

    pub fn device(&self) -> &'d Device {
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
    pub fn upload<D: DataType>(&mut self, data: &[D::HostRepr]) {
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
            .download_kind(self.id, self.kind, self.elem_count)
            .expect("the kind was accepted when this tensor was allocated");
        D::unwrap(data).expect("the device returned the kind it was asked for")
    }

    /// A full copy on `to`, as a new tensor.
    /// Goes through the host for now;
    /// a same-backend device-to-device path comes with integration.
    pub fn copy_to<'e, D: DataType>(&self, to: &'e Device) -> Result<Tensor<'e>, String> {
        self.copy_to_id::<D>(to, TensorId::new())
    }

    /// Like `copy_to`, with the copy's id fixed instead of freshly minted.
    pub(crate) fn copy_to_id<'e, D: DataType>(
        &self,
        to: &'e Device,
        id: TensorId,
    ) -> Result<Tensor<'e>, String> {
        let mut copy = Tensor::with_id(to, id, self.kind, self.elem_count)?;
        copy.upload::<D>(&self.download::<D>());
        Ok(copy)
    }

    /// Writes this tensor into `dst`, starting at element `offset`. Host
    /// round trip for now, like `copy_to`.
    pub fn copy_into<D: DataType>(&self, dst: &mut Tensor, offset: usize) -> Result<(), String> {
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

    // Frees the buffer now, or fails if a built graph still binds it.
    // On failure the tensor is dropped anyway and the buffer goes with the graph.
    pub fn free(self) -> Result<(), String> {
        self.device.check_unused(self.id)
    }
}

impl Drop for Tensor<'_> {
    fn drop(&mut self) {
        self.device.drop_buffer(self.id);
    }
}

#[cfg(test)]
#[path = "../../tests/tensor.rs"]
mod tests;
