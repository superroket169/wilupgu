//! [`TensorSpec`]: a tensor before it's allocated.

use crate::backend::dtype::{DataKind, HostData};
use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{rules, ComputeDevice, Tensor};
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::id::GlobalId;

/// What a tensor holds right after it's built.
#[derive(Debug, Clone)]
pub enum TensorInit {
    /// Unspecified contents, like [`ComputeDevice::alloc`].
    Blank,
    /// This data: of the tensor's kind and length, rounded up to whole host
    /// words for a packed kind.
    Upload(HostData),
    /// All zeros.
    Zero,
}

/// What a [`TensorSpec`] is made from.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// The kind of its elements.
    pub kind: DataKind,
    /// Elements, as in [`ComputeDevice::alloc`].
    pub len: Resolvable<usize>,
    /// What it holds right after it's built.
    pub init: TensorInit,
}

/// A tensor to allocate: its kind, its length, which may be known later, and
/// what it starts with.
#[derive(Debug)]
pub struct TensorSpec {
    id: GlobalId<TensorSpec>,
    desc: TensorDesc,
}

impl TensorSpec {
    /// The kind of its elements.
    pub fn kind(&self) -> DataKind {
        self.desc.kind
    }

    /// Its length in elements, which may be known later.
    pub fn len(&self) -> &Resolvable<usize> {
        &self.desc.len
    }

    /// What it holds right after it's built.
    pub fn init(&self) -> &TensorInit {
        &self.desc.init
    }

    /// Fails if the data to upload is of another kind.
    pub fn new(desc: TensorDesc) -> Result<Self, Error> {
        if let TensorInit::Upload(data) = &desc.init {
            rules::same_kind(desc.kind, data.kind())?;
        }
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    pub fn id(&self) -> GlobalId<TensorSpec> {
        self.id
    }

    /// Fails while the length isn't known, or if the data to upload isn't
    /// of that length, rounded up to whole host words for a packed kind.
    pub fn ready(&self) -> Result<(), Error> {
        let len = *self.desc.len.value()?;
        if let TensorInit::Upload(data) = &self.desc.init {
            let elems = len.next_multiple_of(self.desc.kind.elems_per_host_word());
            rules::same_len(elems, data.len())?;
        }
        Ok(())
    }

    pub fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Tensor<'d, B>, Error> {
        self.ready()?;
        let mut tensor = device.alloc(self.desc.kind, *self.desc.len.value()?)?;
        match &self.desc.init {
            TensorInit::Blank => {}
            TensorInit::Upload(data) => tensor.upload(data)?,
            TensorInit::Zero => tensor.zero()?,
        }
        Ok(tensor)
    }

    /// A [`HollowTensor`] of its kind and length, as many times as needed.
    /// Fails only while the length isn't known; the data to upload isn't
    /// looked at.
    pub fn build_hollow(&self) -> Result<HollowTensor, Error> {
        Ok(HollowTensor {
            kind: self.desc.kind,
            len: *self.desc.len.value()?,
        })
    }
}

/// A tensor that lives in another run: its kind and length, with no
/// memory and no device.
#[derive(Debug)]
pub struct HollowTensor {
    kind: DataKind,
    len: usize,
}

impl HollowTensor {
    /// The kind of its elements.
    pub fn kind(&self) -> DataKind {
        self.kind
    }

    /// Its length in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether its length is 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A tensor of a run: live here, or hollow, living in another run.
pub enum AnyTensor<'d, B: Backend + Topology> {
    /// The tensor itself.
    Live(Tensor<'d, B>),
    /// The place of a tensor that lives in another run.
    Hollow(HollowTensor),
}

impl<'d, B: Backend + Topology> AnyTensor<'d, B> {
    /// The kind of its elements.
    pub fn kind(&self) -> DataKind {
        match self {
            AnyTensor::Live(t) => t.kind(),
            AnyTensor::Hollow(h) => h.kind(),
        }
    }

    /// Its length in elements.
    pub fn len(&self) -> usize {
        match self {
            AnyTensor::Live(t) => t.len(),
            AnyTensor::Hollow(h) => h.len(),
        }
    }

    /// Whether its length is 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether it lives in another run.
    pub fn is_hollow(&self) -> bool {
        matches!(self, AnyTensor::Hollow(_))
    }

    /// The tensor, if it lives here; [`Error::Hollow`] if not.
    pub fn live(&self) -> Result<&Tensor<'d, B>, Error> {
        match self {
            AnyTensor::Live(t) => Ok(t),
            AnyTensor::Hollow(_) => Err(Error::Hollow),
        }
    }

    /// [`AnyTensor::live`], as `&mut`.
    pub fn live_mut(&mut self) -> Result<&mut Tensor<'d, B>, Error> {
        match self {
            AnyTensor::Live(t) => Ok(t),
            AnyTensor::Hollow(_) => Err(Error::Hollow),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/tensor.rs"]
mod tests;
