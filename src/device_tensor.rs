use crate::device::Device;
use crate::specs::TensorId;
use crate::traits::{DataKind, DataType};

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
        let id = TensorId::new();
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
        let copy = Tensor::new(to, self.kind, self.elem_count)?;
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

pub type SplitOp = fn(Tensor, &[(Device, usize)]) -> Result<Vec<Tensor>, String>;
pub type CombineOp = fn(Vec<Tensor>, &[Device]) -> Result<Vec<Tensor>, String>;

/// Tensors on different devices a nicer API for working with several of them at once.
/// the caller picks the `split`/`combine` op that fits.
pub struct SpreadTensor {
    parts: Vec<Tensor>,
}

impl SpreadTensor {
    pub fn new(parts: Vec<Tensor>) -> Result<Self, String> {
        let first = parts
            .first()
            .ok_or("a spread tensor needs at least one part")?;
        for (i, p) in parts.iter().enumerate() {
            if p.kind() != first.kind() {
                return Err(format!(
                    "part {i} is {:?}, part 0 is {:?}",
                    p.kind(),
                    first.kind()
                ));
            }
            if parts[..i]
                .iter()
                .any(|q| q.device().id() == p.device().id())
            {
                return Err(format!("two parts on device {:?}", p.device().id()));
            }
        }
        Ok(Self { parts })
    }

    pub fn kind(&self) -> DataKind {
        self.parts[0].kind()
    }

    pub fn split(tensor: Tensor, to: &[(Device, usize)], op: SplitOp) -> Result<Self, String> {
        Self::new(op(tensor, to)?)
    }

    pub fn combine(self, to: &[Device], op: CombineOp) -> Result<Vec<Tensor>, String> {
        op(self.parts, to)
    }

    pub fn parts(&self) -> &[Tensor] {
        &self.parts
    }

    pub fn into_parts(self) -> Vec<Tensor> {
        self.parts
    }
}

/// Ready-made `SpreadTensor::split` ops.
pub mod split {
    use super::*;

    /// A full copy on each device, every count must be the whole length.
    /// The source becomes the part for its own device if that device is listed,
    /// otherwise it's dropped.
    pub fn replicate<D: DataType>(
        source: Tensor,
        to: &[(Device, usize)],
    ) -> Result<Vec<Tensor>, String> {
        let mut parts = Vec::with_capacity(to.len());
        for (d, n) in to {
            if *n != source.len() {
                return Err(format!("a replica of {n} for a tensor of {}", source.len()));
            }
            if d.id() != source.device().id() {
                parts.push(source.copy_to::<D>(d)?);
            }
        }
        if let Some(i) = to.iter().position(|(d, _)| d.id() == source.device().id()) {
            parts.insert(i, source);
        }
        Ok(parts)
    }

    /// Consecutive slices of the given element counts, one per device, in order.
    pub fn sized<D: DataType>(
        source: Tensor,
        to: &[(Device, usize)],
    ) -> Result<Vec<Tensor>, String> {
        let total: usize = to.iter().map(|(_, n)| n).sum();
        if total != source.len() {
            return Err(format!(
                "slices add up to {total}, the tensor has {}",
                source.len()
            ));
        }
        let per_word = D::ELEMS_PER_HOST_WORD as usize;
        let data = source.download::<D>();
        let mut start = 0;
        let mut parts = Vec::with_capacity(to.len());
        for (d, n) in to {
            if n % per_word != 0 {
                return Err(format!(
                    "slice of {n} isn't a multiple of {per_word}, the packing of {:?}",
                    D::KIND
                ));
            }
            let words = n / per_word;
            let part = Tensor::new(d, D::KIND, *n)?;
            part.upload::<D>(&data[start..start + words]);
            parts.push(part);
            start += words;
        }
        Ok(parts)
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

    fn single(to: &[Device]) -> Result<&Device, String> {
        match to {
            [d] => Ok(d),
            _ => Err(format!("expected one target device, got {}", to.len())),
        }
    }

    /// The parts laid end to end, as one tensor on the single `to` device.
    pub fn concat<D: DataType>(parts: Vec<Tensor>, to: &[Device]) -> Result<Vec<Tensor>, String> {
        let to = single(to)?;
        let total = parts.iter().map(Tensor::len).sum();
        let whole = Tensor::new(to, D::KIND, total)?;
        let mut offset = 0;
        for p in &parts {
            p.copy_into::<D>(&whole, offset)?;
            offset += p.len();
        }
        Ok(vec![whole])
    }

    /// The first part, moved to the single `to` device if it isn't there
    /// already; the rest are dropped. For parts known to be identical copies.
    pub fn take_one<D: DataType>(parts: Vec<Tensor>, to: &[Device]) -> Result<Vec<Tensor>, String> {
        let to = single(to)?;
        let first = parts.into_iter().next().ok_or("nothing to take")?;
        if first.device().id() == to.id() {
            Ok(vec![first])
        } else {
            Ok(vec![first.copy_to::<D>(to)?])
        }
    }
}
