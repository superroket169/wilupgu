//! [`RunPlan`]: tensors, and the calls that run on them, in order. Built, it
//! is a [`Runnable`].

use crate::backend::{Backend, Topology};
use crate::safe::ComputeDevice;
use crate::tools::plan::call::CallSpec;
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::runnable::Runnable;
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::TensorSpec;

/// What a [`RunPlan`] is made from.
pub struct PlanDesc {
    /// Every tensor a call of the plan names.
    pub tensors: Vec<TensorSpec>,
    /// The calls, in the order they run.
    pub calls: Vec<CallSpec>,
}

/// Tensors to build on one device, and the calls that run on them, in order.
pub struct RunPlan {
    id: GlobalId<RunPlan>,
    desc: PlanDesc,
}

impl RunPlan {
    /// The tensors to build.
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.desc.tensors
    }

    /// The calls, in the order they run.
    pub fn calls(&self) -> &[CallSpec] {
        &self.desc.calls
    }
}

impl Spec for RunPlan {
    type Description = PlanDesc;
    type Outcome<'d, B: Backend + Topology + 'd> = Runnable<'d, B>;

    /// Fails if a call names a tensor that isn't in the plan, or one of
    /// another kind than its slot's.
    fn new(desc: PlanDesc) -> Result<Self, String> {
        for call in &desc.calls {
            let shader = call.node().shader();
            for (slot, (id, role)) in call.tensors().iter().zip(shader.layout).enumerate() {
                let tensor = desc.tensors.iter().find(|t| t.id() == *id).ok_or_else(|| {
                    format!(
                        "shader `{}`: slot {} names a tensor that isn't in the plan",
                        shader.name,
                        slot + 1
                    )
                })?;
                if tensor.kind() != role.kind() {
                    return Err(format!(
                        "shader `{}`: slot {} is {:?}, its tensor is {:?}",
                        shader.name,
                        slot + 1,
                        role.kind(),
                        tensor.kind()
                    ));
                }
            }
        }
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    fn id(&self) -> GlobalId<RunPlan> {
        self.id
    }

    /// Fails where a tensor's or call's [`Spec::ready`] does.
    fn ready(&self) -> Result<(), String> {
        for tensor in &self.desc.tensors {
            tensor.ready()?;
        }
        for call in &self.desc.calls {
            call.ready()?;
        }
        Ok(())
    }

    /// Builds every tensor, then every call.
    fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Runnable<'d, B>, String> {
        self.ready()?;
        Runnable::build(self, device)
    }
}

#[cfg(test)]
#[path = "../../tests/run.rs"]
mod tests;
