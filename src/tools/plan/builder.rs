use crate::backend::dtype::{DataKind, HostData};
use crate::backend::shader::{Shader, Workgroups};
use crate::error::Error;
use crate::tools::plan::call::{CallDesc, CallSpec};
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use crate::tools::plan::run::{PlanDesc, RunPlan};
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::{TensorDesc, TensorInit, TensorSpec};

pub struct PlanBuilder {
    tensors: Vec<TensorSpec>,
    calls: Vec<CallSpec>,
}

impl PlanBuilder {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            calls: Vec::new(),
        }
    }

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

    pub fn tensors(&self) -> &[TensorSpec] {
        &self.tensors
    }

    pub fn calls(&self) -> &[CallSpec] {
        &self.calls
    }

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

pub struct TensorBuilder<'p> {
    plan: &'p mut PlanBuilder,
    kind: DataKind,
    len: Resolvable<usize>,
}

impl TensorBuilder<'_> {
    pub fn upload(self, data: HostData) -> Result<GlobalId<TensorSpec>, Error> {
        self.add(TensorInit::Upload(data))
    }

    pub fn zero(self) -> GlobalId<TensorSpec> {
        self.add(TensorInit::Zero).expect("only an upload can fail")
    }

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
