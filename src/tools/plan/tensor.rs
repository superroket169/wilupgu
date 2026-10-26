//! [`TensorSpec`]: a tensor before it's allocated.

use crate::backend::dtype::{DataKind, HostData};
use crate::backend::{Backend, Topology};
use crate::safe::{ComputeDevice, Tensor};
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::spec::Spec;

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
}

impl Spec for TensorSpec {
    type Description = TensorDesc;
    type Outcome<'d, B: Backend + Topology + 'd> = Tensor<'d, B>;

    /// Fails if the data to upload is of another kind.
    fn new(desc: TensorDesc) -> Result<Self, String> {
        if let TensorInit::Upload(data) = &desc.init {
            if data.kind() != desc.kind {
                return Err(format!(
                    "tensor spec of {:?} with {:?} data to upload",
                    desc.kind,
                    data.kind()
                ));
            }
        }
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    fn id(&self) -> GlobalId<TensorSpec> {
        self.id
    }

    /// Fails while the length isn't known, or if the data to upload isn't
    /// of that length, rounded up to whole host words for a packed kind.
    fn ready(&self) -> Result<(), String> {
        let len = *self.desc.len.value()?;
        if let TensorInit::Upload(data) = &self.desc.init {
            let elems = len.next_multiple_of(self.desc.kind.elems_per_host_word());
            if data.len() != elems {
                return Err(format!(
                    "tensor spec of {len} elements with {} to upload",
                    data.len()
                ));
            }
        }
        Ok(())
    }

    fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Tensor<'d, B>, String> {
        self.ready()?;
        let mut tensor = device.alloc(self.desc.kind, *self.desc.len.value()?)?;
        match &self.desc.init {
            TensorInit::Blank => {}
            TensorInit::Upload(data) => tensor.upload(data)?,
            TensorInit::Zero => tensor.zero()?,
        }
        Ok(tensor)
    }
}

#[cfg(test)]
#[path = "../../tests/tensor.rs"]
mod tests;
