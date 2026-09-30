use crate::device::Device;
use crate::specs::{TensorId, TensorSize, TensorSpec};
use crate::traits::{DataKind, DataType, DeviceId, ResolvedSize};

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

/// How a `SpreadTensor` is made: target device + size per part, in order.
/// A plain `fn`, so it's `Copy` and fits in a `SpreadSpec`;
/// the dtype is fixed when it's picked (`split::sized::<F32>`).
pub type SplitOp = fn(Tensor, &[(Device, TensorSize, TensorId)]) -> Result<Vec<Tensor>, String>;

/// How a `SpreadTensor` is ended: the target devices and the whole's id
/// every result tensor gets it. an all-reduce-style combine can leave
/// identical copies on several devices, all still the same logical tensor.
pub type CombineOp = fn(Vec<Tensor>, &[Device], TensorId) -> Result<Vec<Tensor>, String>;

/// A tensor spread across several devices
pub struct SpreadSpec {
    whole: TensorSpec,
    parts: Vec<(DeviceId, TensorSize, TensorId)>,
    split: SplitOp,
    combine: CombineOp,
    combine_to: Vec<DeviceId>,
}

impl SpreadSpec {
    pub fn replicated<D: DataType>(
        whole: TensorSpec,
        devices: Vec<DeviceId>,
        combine_to: DeviceId,
    ) -> Self {
        let parts = devices
            .into_iter()
            .map(|d| (d, whole.size().clone(), TensorId::new()))
            .collect();
        Self {
            whole,
            parts,
            split: split::replicate::<D>,
            combine: combine::take_one::<D>,
            combine_to: vec![combine_to],
        }
    }

    /// Parts sized `unit * multiplier + coefficient`, `unit` shared by all of
    /// them (and by `whole`'s own `Resolvable`); gathered back onto
    /// `combine_to` end to end.
    ///
    /// # Errors
    /// If `whole`'s size isn't `Resolvable`: there's no shared unit to give
    /// the parts if it's `Fixed`.
    pub fn sharded<D: DataType>(
        whole: TensorSpec,
        parts: Vec<(DeviceId, u32, u32)>,
        combine_to: DeviceId,
    ) -> Result<Self, String> {
        let TensorSize::Resolvable(base) = whole.size() else {
            return Err("sharded needs whole's size to be Resolvable".to_string());
        };
        let parts = parts
            .into_iter()
            .map(|(d, multiplier, coefficient)| {
                let size = TensorSize::Resolvable(ResolvedSize {
                    size: base.size.clone(),
                    multiplier,
                    coefficient,
                });
                (d, size, TensorId::new())
            })
            .collect();
        Ok(Self {
            whole,
            parts,
            split: split::sized::<D>,
            combine: combine::concat::<D>,
            combine_to: vec![combine_to],
        })
    }

    /// Any split/combine op, for cases the ready-made constructors don't cover.
    pub fn custom(
        whole: TensorSpec,
        parts: Vec<(DeviceId, TensorSize)>,
        split: SplitOp,
        combine: CombineOp,
        combine_to: Vec<DeviceId>,
    ) -> Self {
        let parts = parts
            .into_iter()
            .map(|(d, s)| (d, s, TensorId::new()))
            .collect();
        Self {
            whole,
            parts,
            split,
            combine,
            combine_to,
        }
    }

    pub fn whole_id(&self) -> TensorId {
        self.whole.id()
    }

    pub fn whole(&self) -> &TensorSpec {
        &self.whole
    }

    pub fn part_id(&self, device: DeviceId) -> Option<TensorId> {
        self.parts
            .iter()
            .find(|(d, ..)| *d == device)
            .map(|(_, _, id)| *id)
    }

    pub fn part_ids(&self) -> Vec<(DeviceId, TensorId)> {
        self.parts.iter().map(|(d, _, id)| (*d, *id)).collect()
    }

    pub fn combine_to(&self) -> &[DeviceId] {
        &self.combine_to
    }

    pub(crate) fn parts(&self) -> &[(DeviceId, TensorSize, TensorId)] {
        &self.parts
    }

    pub(crate) fn split_op(&self) -> SplitOp {
        self.split
    }

    pub(crate) fn combine_op(&self) -> CombineOp {
        self.combine
    }
}

pub struct SpreadTensor {
    whole_id: TensorId,
    part_ids: Vec<TensorId>,
    state: SpreadState,
}

enum SpreadState {
    Whole(Tensor),
    Parts(Vec<Tensor>),
}

impl SpreadTensor {
    /// Wraps `tensor` as `spec`'s whole side; the parts stay shadow until `split`.
    ///
    /// # Panics
    /// If `tensor`'s kind doesn't match `spec`'s.
    pub fn from_whole(tensor: Tensor, spec: &SpreadSpec) -> Self {
        assert_eq!(
            tensor.kind(),
            spec.whole().kind(),
            "tensor holds {:?}, spec expects {:?}",
            tensor.kind(),
            spec.whole().kind()
        );
        Self {
            whole_id: spec.whole_id(),
            part_ids: spec.part_ids().into_iter().map(|(_, id)| id).collect(),
            state: SpreadState::Whole(tensor),
        }
    }

    pub fn whole_id(&self) -> TensorId {
        self.whole_id
    }

    pub fn part_ids(&self) -> &[TensorId] {
        &self.part_ids
    }

    /// The live whole, if this side is combined; `None` while it's split.
    pub fn whole(&self) -> Option<&Tensor> {
        match &self.state {
            SpreadState::Whole(t) => Some(t),
            SpreadState::Parts(_) => None,
        }
    }

    /// The live parts, if this side is split; `None` while it's whole.
    pub fn parts(&self) -> Option<&[Tensor]> {
        match &self.state {
            SpreadState::Parts(p) => Some(p),
            SpreadState::Whole(_) => None,
        }
    }

    /// Resolves `spec`'s devices against `devices`, splits the whole side into parts.
    ///
    /// # Errors
    /// Already split, or a device in `spec` isn't in `devices`.
    pub fn split(&mut self, spec: &SpreadSpec, devices: &[Device]) -> Result<(), String> {
        if matches!(self.state, SpreadState::Parts(_)) {
            return Err("already split".to_string());
        }
        let mut targets = Vec::with_capacity(spec.parts().len());
        for (device_id, size, id) in spec.parts() {
            targets.push((
                resolve_device(devices, *device_id)?.clone(),
                size.clone(),
                *id,
            ));
        }
        let SpreadState::Whole(whole) =
            std::mem::replace(&mut self.state, SpreadState::Parts(Vec::new()))
        else {
            unreachable!()
        };

        // on error, `whole` was already consumed (and dropped) inside the op:
        // this spread tensor is left holding no buffers at all
        self.state = SpreadState::Parts(spec.split_op()(whole, &targets)?);
        Ok(())
    }

    /// The reverse of `split`.
    ///
    /// # Errors
    /// Already whole, or a device in `spec` isn't in `devices`.
    pub fn combine(&mut self, spec: &SpreadSpec, devices: &[Device]) -> Result<(), String> {
        if matches!(self.state, SpreadState::Whole(_)) {
            return Err("already whole".to_string());
        }
        let mut to = Vec::with_capacity(spec.combine_to().len());
        for device_id in spec.combine_to() {
            to.push(resolve_device(devices, *device_id)?.clone());
        }
        let SpreadState::Parts(parts) =
            std::mem::replace(&mut self.state, SpreadState::Parts(Vec::new()))
        else {
            unreachable!()
        };
        let mut result = spec.combine_op()(parts, &to, spec.whole_id())?;
        self.state = SpreadState::Whole(result.pop().ok_or("combine produced no tensor")?);
        Ok(())
    }
}

fn resolve_device(devices: &[Device], id: DeviceId) -> Result<&Device, String> {
    devices
        .iter()
        .find(|d| d.id() == id)
        .ok_or_else(|| format!("no device for {id:?}"))
}

/// Ready-made `SpreadSpec::split` ops.
pub mod split {
    use super::*;

    /// A full copy on each device, every part's size must be the whole length.
    pub fn replicate<D: DataType>(
        source: Tensor,
        to: &[(Device, TensorSize, TensorId)],
    ) -> Result<Vec<Tensor>, String> {
        let mut parts = Vec::with_capacity(to.len());
        for (d, size, id) in to {
            let n = size.resolved() as usize;
            if n != source.len() {
                return Err(format!("a replica of {n} for a tensor of {}", source.len()));
            }
            parts.push(source.copy_to_id::<D>(d, *id)?);
        }
        Ok(parts)
    }

    /// Consecutive slices, one per device, in order.
    pub fn sized<D: DataType>(
        source: Tensor,
        to: &[(Device, TensorSize, TensorId)],
    ) -> Result<Vec<Tensor>, String> {
        let total: usize = to.iter().map(|(_, s, _)| s.resolved() as usize).sum();
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
        for (d, size, id) in to {
            let n = size.resolved() as usize;
            if n % per_word != 0 {
                return Err(format!(
                    "slice of {n} isn't a multiple of {per_word}, the packing of {:?}",
                    D::KIND
                ));
            }
            let words = n / per_word;
            let part = Tensor::with_id(d, *id, D::KIND, n)?;
            part.upload::<D>(&data[start..start + words]);
            parts.push(part);
            start += words;
        }
        Ok(parts)
    }
}

/// Ready-made `SpreadSpec::combine` ops.
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
    pub fn concat<D: DataType>(
        parts: Vec<Tensor>,
        to: &[Device],
        whole_id: TensorId,
    ) -> Result<Vec<Tensor>, String> {
        let to = single(to)?;
        let total = parts.iter().map(Tensor::len).sum();
        let whole = Tensor::with_id(to, whole_id, D::KIND, total)?;
        let mut offset = 0;
        for p in &parts {
            p.copy_into::<D>(&whole, offset)?;
            offset += p.len();
        }
        Ok(vec![whole])
    }

    /// The first part, moved to the single `to` device (and given `whole_id`)
    /// even if it's already there. For parts known to be identical copies.
    pub fn take_one<D: DataType>(
        parts: Vec<Tensor>,
        to: &[Device],
        whole_id: TensorId,
    ) -> Result<Vec<Tensor>, String> {
        let to = single(to)?;
        let first = parts.into_iter().next().ok_or("nothing to take")?;
        Ok(vec![first.copy_to_id::<D>(to, whole_id)?])
    }
}
