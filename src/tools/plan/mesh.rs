//! [`MeshSpec`]: tensors, and routines of calls on them, each run by its key.

use std::collections::HashSet;

use crate::error::Error;
use crate::tools::plan::call::CallSpec;
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
}

#[cfg(test)]
#[path = "../../tests/mesh.rs"]
mod tests;
