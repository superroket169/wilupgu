//! [`PlanBuilder`]: a [`RunPlan`] made a step at a time. Each step makes the
//! spec it stands for and keeps it; ids come back to name tensors in calls.

use crate::backend::dtype::{DataKind, HostData};
use crate::backend::shader::{Shader, Workgroups};
use crate::error::Error;
use crate::tools::plan::call::{CallDesc, CallSpec};
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use crate::tools::plan::run::{PlanDesc, RunPlan};
use crate::tools::plan::tensor::{TensorDesc, TensorInit, TensorSpec};

/// The tensors and calls of a [`RunPlan`] to come, in the order they were
/// added.
pub struct PlanBuilder {
    tensors: Vec<TensorSpec>,
    calls: Vec<CallSpec>,
}

impl PlanBuilder {
    /// An empty plan.
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            calls: Vec::new(),
        }
    }

    /// A tensor of `kind` and `len` elements; the [`TensorBuilder`] adds it
    /// once it is told what the tensor starts with.
    pub fn tensor(
        &mut self,
        kind: DataKind,
        len: impl Into<Resolvable<usize>>,
    ) -> TensorBuilder<'_> {
        TensorBuilder {
            plan: self,
            kind,
            len: len.into(),
        }
    }

    /// A call of `shader` on `tensors`, one per slot in slot order, with its
    /// node: `meta`, one per field of the shader's meta, and `workgroups`.
    ///
    /// Fails where [`NodeSpec::new`] or [`CallSpec::new`] does; nothing is
    /// added then.
    pub fn call(
        &mut self,
        shader: Shader,
        meta: Vec<MetaSource>,
        workgroups: impl Into<Resolvable<Workgroups>>,
        tensors: &[GlobalId<TensorSpec>],
    ) -> Result<GlobalId<CallSpec>, Error> {
        let node = NodeSpec::new(NodeDesc {
            shader,
            meta,
            workgroups: workgroups.into(),
        })?;
        let call = CallSpec::new(CallDesc {
            node,
            tensors: tensors.to_vec(),
        })?;
        let id = call.id();
        self.calls.push(call);
        Ok(id)
    }

    /// The tensors added so far.
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.tensors
    }

    /// The calls added so far, in the order they run.
    pub fn calls(&self) -> &[CallSpec] {
        &self.calls
    }

    /// The [`RunPlan`] of everything added. Fails where [`RunPlan::new`]
    /// does, like for a call that names a tensor of another builder.
    pub fn finish(self) -> Result<RunPlan, Error> {
        RunPlan::new(PlanDesc {
            tensors: self.tensors,
            calls: self.calls,
        })
    }
}

impl Default for PlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A tensor of [`PlanBuilder::tensor`], waiting for what it starts with.
/// Each method adds it to the plan and returns its id.
pub struct TensorBuilder<'p> {
    plan: &'p mut PlanBuilder,
    kind: DataKind,
    len: Resolvable<usize>,
}

impl TensorBuilder<'_> {
    /// Starts with `data`: of the tensor's kind and length, rounded up to
    /// whole host words for a packed kind.
    ///
    /// Fails if `data` is of another kind; nothing is added then. A length
    /// that doesn't match fails when the plan is built, since the length
    /// may be known only then.
    pub fn upload(self, data: HostData) -> Result<GlobalId<TensorSpec>, Error> {
        self.add(TensorInit::Upload(data))
    }

    /// Starts with all zeros.
    pub fn zero(self) -> GlobalId<TensorSpec> {
        self.add(TensorInit::Zero).expect("only an upload can fail")
    }

    /// Starts with unspecified contents, like [`ComputeDevice::alloc`](crate::safe::ComputeDevice::alloc).
    pub fn blank(self) -> GlobalId<TensorSpec> {
        self.add(TensorInit::Blank)
            .expect("only an upload can fail")
    }

    fn add(self, init: TensorInit) -> Result<GlobalId<TensorSpec>, Error> {
        let tensor = TensorSpec::new(TensorDesc {
            kind: self.kind,
            len: self.len,
            init,
        })?;
        let id = tensor.id();
        self.plan.tensors.push(tensor);
        Ok(id)
    }
}

#[cfg(all(test, feature = "tool-builtins"))]
#[path = "../../tests/builder.rs"]
mod tests;
