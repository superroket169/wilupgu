//! A tensor spread across several devices: the up-front plan (`SpreadSpec`,
//! every id decided before anything exists) and the live, two-state handle
//! (`SpreadTensor`) -- whichever side is live holds real buffers, the other
//! is a pure shadow of ids.

use crate::backend::dtype::DataType;
use crate::tools::core::deferred::Resolvable;
use crate::tools::core::device::Device;
use crate::tools::core::id::{DeviceId, TensorId};
use crate::tools::core::tensor::{Tensor, TensorSpec};

/// How a `SpreadTensor` is made: target device + size per part, in order.
/// A plain `fn`, so it's `Copy` and fits in a `SpreadSpec`;
/// the dtype is fixed when it's picked (`split::sized::<F32>`).
pub type SplitOp = for<'d> fn(
    Tensor<'d>,
    &[(&'d Device, Resolvable<u32>, TensorId)],
) -> Result<Vec<Tensor<'d>>, String>;

/// How a `SpreadTensor` is ended: the target devices and the whole's id
/// every result tensor gets it. an all-reduce-style combine can leave
/// identical copies on several devices, all still the same logical tensor.
pub type CombineOp =
    for<'d> fn(Vec<Tensor<'d>>, &[&'d Device], TensorId) -> Result<Vec<Tensor<'d>>, String>;

/// A tensor spread across several devices
pub struct SpreadSpec {
    whole: TensorSpec,
    parts: Vec<(DeviceId, Resolvable<u32>, TensorId)>,
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

    /// Gathered back onto `combine_to` end to end.
    pub fn sharded<D: DataType>(
        whole: TensorSpec,
        parts: Vec<(DeviceId, Resolvable<u32>)>,
        combine_to: DeviceId,
    ) -> Self {
        let parts = parts
            .into_iter()
            .map(|(d, size)| (d, size, TensorId::new()))
            .collect();
        Self {
            whole,
            parts,
            split: split::sized::<D>,
            combine: combine::concat::<D>,
            combine_to: vec![combine_to],
        }
    }

    /// Any split/combine op, for cases the ready-made constructors don't cover.
    pub fn custom(
        whole: TensorSpec,
        parts: Vec<(DeviceId, Resolvable<u32>)>,
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

    pub(crate) fn parts(&self) -> &[(DeviceId, Resolvable<u32>, TensorId)] {
        &self.parts
    }

    pub(crate) fn split_op(&self) -> SplitOp {
        self.split
    }

    pub(crate) fn combine_op(&self) -> CombineOp {
        self.combine
    }
}

/// Tensors on different devices, seen as one. Whichever side is live (the
/// whole or the parts) holds real buffers; the other is a shadow -- just the
/// ids it would use if it existed.
pub struct SpreadTensor<'d> {
    whole_id: TensorId,
    part_ids: Vec<TensorId>,
    state: SpreadState<'d>,
}

enum SpreadState<'d> {
    Whole(Tensor<'d>),
    Parts(Vec<Tensor<'d>>),
}

impl<'d> SpreadTensor<'d> {
    /// Wraps `tensor` as `spec`'s whole side; the parts stay shadow until `split`.
    ///
    /// # Panics
    /// If `tensor`'s kind doesn't match `spec`'s.
    pub fn from_whole(tensor: Tensor<'d>, spec: &SpreadSpec) -> Self {
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
    pub fn whole(&self) -> Option<&Tensor<'d>> {
        match &self.state {
            SpreadState::Whole(t) => Some(t),
            SpreadState::Parts(_) => None,
        }
    }

    /// The live parts, if this side is split; `None` while it's whole.
    pub fn parts(&self) -> Option<&[Tensor<'d>]> {
        match &self.state {
            SpreadState::Parts(p) => Some(p),
            SpreadState::Whole(_) => None,
        }
    }

    /// Resolves `spec`'s devices against `devices`, splits the whole side into parts.
    ///
    /// # Errors
    /// Already split, or a device in `spec` isn't in `devices`.
    pub fn split(&mut self, spec: &SpreadSpec, devices: &'d [Device]) -> Result<(), String> {
        if matches!(self.state, SpreadState::Parts(_)) {
            return Err("already split".to_string());
        }
        let mut targets = Vec::with_capacity(spec.parts().len());
        for (device_id, size, id) in spec.parts() {
            targets.push((resolve_device(devices, *device_id)?, size.clone(), *id));
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
    pub fn combine(&mut self, spec: &SpreadSpec, devices: &'d [Device]) -> Result<(), String> {
        if matches!(self.state, SpreadState::Whole(_)) {
            return Err("already whole".to_string());
        }
        let mut to = Vec::with_capacity(spec.combine_to().len());
        for device_id in spec.combine_to() {
            to.push(resolve_device(devices, *device_id)?);
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
    pub fn replicate<'d, D: DataType>(
        source: Tensor<'d>,
        to: &[(&'d Device, Resolvable<u32>, TensorId)],
    ) -> Result<Vec<Tensor<'d>>, String> {
        let mut parts = Vec::with_capacity(to.len());
        for (d, size, id) in to {
            let n = *size.value() as usize;
            if n != source.len() {
                return Err(format!("a replica of {n} for a tensor of {}", source.len()));
            }
            parts.push(source.copy_to_id::<D>(*d, *id)?);
        }
        Ok(parts)
    }

    /// Consecutive slices, one per device, in order.
    pub fn sized<'d, D: DataType>(
        source: Tensor<'d>,
        to: &[(&'d Device, Resolvable<u32>, TensorId)],
    ) -> Result<Vec<Tensor<'d>>, String> {
        let total: usize = to.iter().map(|(_, s, _)| *s.value() as usize).sum();
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
            let n = *size.value() as usize;
            if n % per_word != 0 {
                return Err(format!(
                    "slice of {n} isn't a multiple of {per_word}, the packing of {:?}",
                    D::KIND
                ));
            }
            let words = n / per_word;
            let mut part = Tensor::with_id(*d, *id, D::KIND, n)?;
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

    fn single<'d>(to: &[&'d Device]) -> Result<&'d Device, String> {
        match to {
            [d] => Ok(*d),
            _ => Err(format!("expected one target device, got {}", to.len())),
        }
    }

    /// The parts laid end to end, as one tensor on the single `to` device.
    pub fn concat<'d, D: DataType>(
        parts: Vec<Tensor<'d>>,
        to: &[&'d Device],
        whole_id: TensorId,
    ) -> Result<Vec<Tensor<'d>>, String> {
        let to = single(to)?;
        let total = parts.iter().map(Tensor::len).sum();
        let mut whole = Tensor::with_id(to, whole_id, D::KIND, total)?;
        let mut offset = 0;
        for p in &parts {
            p.copy_into::<D>(&mut whole, offset)?;
            offset += p.len();
        }
        Ok(vec![whole])
    }

    /// The first part, moved to the single `to` device (and given `whole_id`)
    /// even if it's already there. For parts known to be identical copies.
    pub fn take_one<'d, D: DataType>(
        parts: Vec<Tensor<'d>>,
        to: &[&'d Device],
        whole_id: TensorId,
    ) -> Result<Vec<Tensor<'d>>, String> {
        let to = single(to)?;
        let first = parts.into_iter().next().ok_or("nothing to take")?;
        Ok(vec![first.copy_to_id::<D>(to, whole_id)?])
    }
}
