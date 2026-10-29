//! [`RunSpec`]: tensors, and the calls that run on them, in order. Built, it
//! is a [`Run`].

use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{rules, ComputeDevice};
use crate::tools::plan::call::CallSpec;
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::runnable::Run;
use crate::tools::plan::tensor::TensorSpec;

/// What a [`RunSpec`] is made from.
pub struct RunDesc {
    /// Every tensor a call of the plan names.
    pub tensors: Vec<TensorSpec>,
    /// The calls, in the order they run.
    pub calls: Vec<CallSpec>,
}

/// Tensors to build on one device, and the calls that run on them, in order.
pub struct RunSpec {
    id: GlobalId<RunSpec>,
    desc: RunDesc,
}

impl RunSpec {
    /// The tensors to build.
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.desc.tensors
    }

    /// The calls, in the order they run.
    pub fn calls(&self) -> &[CallSpec] {
        &self.desc.calls
    }

    /// Fails if a call names a tensor that isn't in the plan, one of another
    /// kind than its slot's, or a written slot's tensor in another slot too.
    pub fn new(desc: RunDesc) -> Result<Self, Error> {
        for call in &desc.calls {
            let kinds = call
                .tensors()
                .iter()
                .map(|id| {
                    let tensor = desc.tensors.iter().find(|t| t.id() == *id);
                    Ok(tensor.ok_or(Error::NotInPlan { what: "tensor" })?.kind())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            rules::slot_kinds(call.node().shader().signature(), kinds.into_iter())?;
            rules::not_aliased(call.node().shader().signature(), call.tensors())?;
        }
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    pub fn id(&self) -> GlobalId<RunSpec> {
        self.id
    }

    /// Fails where [`TensorSpec::ready`] or [`CallSpec::ready`] does.
    pub fn ready(&self) -> Result<(), Error> {
        for tensor in &self.desc.tensors {
            tensor.ready()?;
        }
        for call in &self.desc.calls {
            call.ready()?;
        }
        Ok(())
    }

    /// Builds every tensor, then every call. Also fails if a tensor is
    /// shorter than its slot needs for its call's meta.
    pub fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Run<'d, B>, Error> {
        self.ready()?;
        Run::build(self, device)
    }
}

#[cfg(test)]
#[path = "../../tests/run.rs"]
mod tests;
