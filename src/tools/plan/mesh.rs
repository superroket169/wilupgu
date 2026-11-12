//! [`MeshSpec`]: tensors, and routines of calls on them, each run by its key.

use std::collections::{HashMap, HashSet};

use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{rules, Arg, ComputeDevice, NodeSlot, Scope, Tensor, TensorSlot};
use crate::tools::plan::call::{Call, CallSpec};
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::run::{bind_call, check_calls, read_dynamics, PerRun};
use crate::tools::plan::tensor::TensorSpec;

/// The tag of a routine's key, `GlobalId<Routine>`.
pub enum Routine {}

impl Routine {
    pub fn key() -> GlobalId<Routine> {
        GlobalId::new()
    }
}

/// What a [`MeshSpec`] is made from.
pub struct MeshDesc {
    pub tensors: Vec<TensorSpec>,
    pub routines: Vec<(GlobalId<Routine>, Vec<CallSpec>)>,
}

pub struct MeshSpec {
    desc: MeshDesc,
}

impl MeshSpec {
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.desc.tensors
    }

    pub fn routines(&self) -> &[(GlobalId<Routine>, Vec<CallSpec>)] {
        &self.desc.routines
    }

    pub fn routine(&self, key: GlobalId<Routine>) -> Result<&[CallSpec], Error> {
        self.desc
            .routines
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, calls)| calls.as_slice())
            .ok_or(Error::NotInPlan { what: "routine" })
    }

    /// Gives the desc back, to change and check again with [`MeshSpec::new`].
    pub fn into_desc(self) -> MeshDesc {
        self.desc
    }

    pub fn new(desc: MeshDesc) -> Result<Self, Error> {
        let mut keys = HashSet::new();
        for (key, calls) in &desc.routines {
            if !keys.insert(*key) {
                return Err(Error::DuplicateId { what: "routine" });
            }
            check_calls(&desc.tensors, calls)?;
        }
        Ok(Self { desc })
    }

    /// Fails where [`TensorSpec::ready`] or [`CallSpec::ready`] does.
    pub fn ready(&self) -> Result<(), Error> {
        for tensor in &self.desc.tensors {
            tensor.ready()?;
        }
        for call in self.desc.routines.iter().flat_map(|(_, calls)| calls) {
            call.ready()?;
        }
        Ok(())
    }

    pub fn build<'d, B: Backend + Topology>(
        self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Mesh<'d, B>, (Self, Error)> {
        match self.ready().and_then(|()| Mesh::build(&self, device)) {
            Ok(mesh) => Ok(mesh),
            Err(e) => Err((self, e)),
        }
    }
}

pub struct Mesh<'d, B: Backend + Topology> {
    device: &'d ComputeDevice<B>,
    tensors: HashMap<GlobalId<TensorSpec>, Tensor<'d, B>>,
    /// Each routine's calls, in the order they run.
    routines: HashMap<GlobalId<Routine>, Vec<Call<'d, B>>>,
}

impl<'d, B: Backend + Topology> Mesh<'d, B> {
    /// `MeshSpec::build`, once the spec is ready.
    fn build(spec: &MeshSpec, device: &'d ComputeDevice<B>) -> Result<Self, Error> {
        let mesh = Self {
            device,
            tensors: spec
                .tensors()
                .iter()
                .map(|t| Ok((t.id(), t.build(device)?)))
                .collect::<Result<_, Error>>()?,
            routines: spec
                .routines()
                .iter()
                .map(|(key, calls)| {
                    let calls = calls
                        .iter()
                        .map(|c| c.build(device))
                        .collect::<Result<_, Error>>()?;
                    Ok((*key, calls))
                })
                .collect::<Result<_, Error>>()?,
        };
        for call in mesh.routines.values().flatten() {
            let shader = call.node().signature();
            for (slot, (&id, &needs)) in call
                .tensors()
                .iter()
                .zip(call.node().min_lens())
                .enumerate()
            {
                rules::long_enough(shader, slot, needs, mesh.tensor(id)?.len())?;
            }
        }
        Ok(mesh)
    }

    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.device
    }

    pub fn tensors(&self) -> impl Iterator<Item = (GlobalId<TensorSpec>, &Tensor<'d, B>)> {
        self.tensors.iter().map(|(&id, tensor)| (id, tensor))
    }

    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<&Tensor<'d, B>, Error> {
        self.tensors
            .get(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })
    }

    pub fn routines(&self) -> impl Iterator<Item = (GlobalId<Routine>, &[Call<'d, B>])> {
        self.routines
            .iter()
            .map(|(&key, calls)| (key, calls.as_slice()))
    }

    pub fn routine(&self, key: GlobalId<Routine>) -> Result<&[Call<'d, B>], Error> {
        self.routines
            .get(&key)
            .map(Vec::as_slice)
            .ok_or(Error::NotInPlan { what: "routine" })
    }

    /// Binds every tensor and node into `s` until it ends; the handle runs
    /// each routine by its key, as many times as needed.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<MeshBound<'scope, 'env, 's, B>, Error> {
        let tensors: HashMap<GlobalId<TensorSpec>, TensorSlot<'scope>> = self
            .tensors
            .iter_mut()
            .map(|(&id, tensor)| Ok((id, s.bind(tensor)?)))
            .collect::<Result<_, Error>>()?;
        let mut routines = HashMap::with_capacity(self.routines.len());
        let mut per_run = Vec::new();
        for (&key, calls) in self.routines.iter_mut() {
            let calls = calls
                .iter_mut()
                .map(|call| bind_call(s, call, &tensors, &mut per_run))
                .collect::<Result<_, Error>>()?;
            routines.insert(key, calls);
        }
        Ok(MeshBound {
            scope: s,
            tensors,
            routines,
            per_run,
        })
    }
}

/// A [`Mesh`] bound into a scope by [`Mesh::bind`].
pub struct MeshBound<'scope, 'env, 's, B: Backend + Topology> {
    scope: &'scope Scope<'scope, 'env, 's, B>,
    tensors: HashMap<GlobalId<TensorSpec>, TensorSlot<'scope>>,
    /// Each routine's calls, in order: the node and its args.
    routines: HashMap<GlobalId<Routine>, Vec<(NodeSlot<'scope>, Vec<Arg<'scope>>)>>,
    per_run: PerRun<'scope>,
}

impl<'scope, B: Backend + Topology> MeshBound<'scope, '_, '_, B> {
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<TensorSlot<'scope>, Error> {
        Ok(*self
            .tensors
            .get(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?)
    }

    /// Queues the calls of the routine with `key`, in order. A call that
    /// fails its checks stops it with [`Error::InCall`], naming its place in
    /// the routine; the calls before it are queued.
    pub fn execute(&self, key: GlobalId<Routine>) -> Result<(), Error> {
        let calls = self
            .routines
            .get(&key)
            .ok_or(Error::NotInPlan { what: "routine" })?;
        for (index, (node, args)) in calls.iter().enumerate() {
            self.scope
                .execute(*node, args)
                .map_err(|error| Error::InCall {
                    index,
                    error: Box::new(error),
                })?;
        }
        Ok(())
    }

    /// Like [`Bound::apply_dynamics`](super::run::Bound::apply_dynamics),
    /// for every routine.
    pub fn apply_dynamics(&self) -> Result<(), Error> {
        for (node, meta) in read_dynamics(&self.per_run)? {
            self.scope.update_meta(node, &meta)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/mesh.rs"]
mod tests;
