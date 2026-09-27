use std::marker::PhantomData;

use crate::device::Device;
use crate::specs::TensorId;
use crate::traits::DataType;

/// A live tensor on one device: just a handle (device + id)
/// the buffer itself lives in the device's own table.
///
/// Not `Clone`
pub struct Tensor<D: DataType> {
    device: Device,
    id: TensorId,
    elem_count: usize,
    _d: PhantomData<D>,
}

impl<D: DataType> Tensor<D> {
    pub fn new(device: &Device, elem_count: usize) -> Result<Self, String> {
        let id = TensorId::new();
        device.alloc_kind(id, D::KIND, elem_count)?;
        Ok(Self {
            device: device.clone(),
            id,
            elem_count,
            _d: PhantomData,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn len(&self) -> usize {
        self.elem_count
    }

    /// Host words this tensor spans
    ///
    /// differs from `len` only for packed
    /// kinds (Int4: 8 values per word).
    fn host_len(&self) -> usize {
        self.elem_count.div_ceil(D::ELEMS_PER_HOST_WORD as usize)
    }

    pub fn upload(&self, data: &[D::HostRepr]) -> Result<(), String> {
        if data.len() != self.host_len() {
            return Err(format!(
                "upload of {} host words into a tensor of {}",
                data.len(),
                self.host_len()
            ));
        }
        self.device.upload_kind(self.id, &D::wrap(data.to_vec()))
    }

    pub fn download(&self) -> Vec<D::HostRepr> {
        let data = self
            .device
            .download_kind(self.id, D::KIND)
            .expect("the kind was accepted when this tensor was allocated");
        D::unwrap(data).expect("the device returned the kind it was asked for")
    }

    /// A full copy on `to`, as a new tensor.
    /// Goes through the host for now;
    /// a same-backend device-to-device path comes with integration.
    pub fn copy_to(&self, to: &Device) -> Result<Tensor<D>, String> {
        let copy = Tensor::new(to, self.elem_count)?;
        copy.upload(&self.download())?;
        Ok(copy)
    }

    /// Writes this tensor into `dst`, starting at element `offset`. Host
    /// round trip for now, like `copy_to`.
    pub fn copy_into(&self, dst: &Tensor<D>, offset: usize) -> Result<(), String> {
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
        let mut whole = dst.download();
        let start = offset / per_word;
        let part = self.download();
        whole[start..start + part.len()].copy_from_slice(&part);
        dst.upload(&whole)
    }
}

impl<D: DataType> Drop for Tensor<D> {
    fn drop(&mut self) {
        self.device.drop_buffer(self.id);
    }
}

/// Tensors on different devices a nicer API for working with several of them at once.
/// the caller picks the `split`/`combine` op that fits.
pub struct SpreadTensor<D: DataType> {
    parts: Vec<Tensor<D>>,
}

impl<D: DataType> SpreadTensor<D> {
    pub fn new(parts: Vec<Tensor<D>>) -> Result<Self, String> {
        for (i, p) in parts.iter().enumerate() {
            if parts[..i]
                .iter()
                .any(|q| q.device().id() == p.device().id())
            {
                return Err(format!("two parts on device {:?}", p.device().id()));
            }
        }
        Ok(Self { parts })
    }

    pub fn split(
        tensor: Tensor<D>,
        op: impl Fn(Tensor<D>) -> Result<Vec<Tensor<D>>, String>,
    ) -> Result<Self, String> {
        Self::new(op(tensor)?)
    }

    pub fn combine(
        self,
        op: impl Fn(Vec<Tensor<D>>) -> Result<Vec<Tensor<D>>, String>,
    ) -> Result<Vec<Tensor<D>>, String> {
        op(self.parts)
    }

    pub fn parts(&self) -> &[Tensor<D>] {
        &self.parts
    }

    pub fn into_parts(self) -> Vec<Tensor<D>> {
        self.parts
    }
}

/// Ready-made `SpreadTensor::split` ops.
pub mod split {
    use super::*;

    /// A full copy on each device. The source becomes the part for its own
    /// device if that device is listed, otherwise it's dropped.
    pub fn replicate<D: DataType>(
        to: &[Device],
    ) -> impl Fn(Tensor<D>) -> Result<Vec<Tensor<D>>, String> {
        let to = to.to_vec();
        move |source| {
            let mut parts = Vec::with_capacity(to.len());
            for d in &to {
                if d.id() != source.device().id() {
                    parts.push(source.copy_to(d)?);
                }
            }
            if let Some(i) = to.iter().position(|d| d.id() == source.device().id()) {
                parts.insert(i, source);
            }
            Ok(parts)
        }
    }

    /// Consecutive slices of the given element counts, one per device, in order.
    pub fn sized<D: DataType>(
        to: &[(Device, usize)],
    ) -> impl Fn(Tensor<D>) -> Result<Vec<Tensor<D>>, String> {
        let to = to.to_vec();
        move |source| {
            let total: usize = to.iter().map(|(_, n)| n).sum();
            if total != source.len() {
                return Err(format!(
                    "slices add up to {total}, the tensor has {}",
                    source.len()
                ));
            }
            let per_word = D::ELEMS_PER_HOST_WORD as usize;
            let data = source.download();
            let mut start = 0;
            let mut parts = Vec::with_capacity(to.len());
            for (d, n) in &to {
                if n % per_word != 0 {
                    return Err(format!(
                        "slice of {n} isn't a multiple of {per_word}, the packing of {:?}",
                        D::KIND
                    ));
                }
                let words = n / per_word;
                let part = Tensor::new(d, *n)?;
                part.upload(&data[start..start + words])?;
                parts.push(part);
                start += words;
            }
            Ok(parts)
        }
    }
}

/// Ready-made `SpreadTensor::combine` ops.
///
/// NOTE:
/// `sum`/`sum_to_all` are missing:
/// they need an on-device add shader run through `Device::run_once`, and the
/// built-in shaders move to the new `Shader` type only with integration.
pub mod combine {
    use super::*;

    /// The parts laid end to end, as one tensor on `to`.
    pub fn concat<D: DataType>(
        to: &Device,
    ) -> impl Fn(Vec<Tensor<D>>) -> Result<Vec<Tensor<D>>, String> {
        let to = to.clone();
        move |parts| {
            let total = parts.iter().map(Tensor::len).sum();
            let whole = Tensor::new(&to, total)?;
            let mut offset = 0;
            for p in &parts {
                p.copy_into(&whole, offset)?;
                offset += p.len();
            }
            Ok(vec![whole])
        }
    }

    /// The first part, moved to `to` if it isn't there already; the rest are
    /// dropped. For parts known to be identical copies.
    pub fn take_one<D: DataType>(
        to: &Device,
    ) -> impl Fn(Vec<Tensor<D>>) -> Result<Vec<Tensor<D>>, String> {
        let to = to.clone();
        move |parts| {
            let first = parts.into_iter().next().ok_or("nothing to take")?;
            if first.device().id() == to.id() {
                Ok(vec![first])
            } else {
                Ok(vec![first.copy_to(&to)?])
            }
        }
    }
}
