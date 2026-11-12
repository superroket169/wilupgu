//! [`MeshSpec`]: tensors, and routines of calls on them, each run by its key.

use std::collections::{HashMap, HashSet};

use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{rules, ComputeDevice, Tensor};
use crate::tools::plan::call::{Call, CallSpec};
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::run::check_calls;
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
}

#[cfg(test)]
#[path = "../../tests/mesh.rs"]
mod tests;
