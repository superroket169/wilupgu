//! The checks made in more than one place, each one a function returning
//! the [`Error`] it finds. They take plain values, so the safe layer and the
//! tools share them.

use crate::backend::dtype::DataKind;
use crate::backend::shader::{BindingRole, MetaType, Shader, Signature, Workgroups};
use crate::backend::{Backend, DeviceInfo as _, Topology};
use crate::error::Error;

use super::ComputeDevice;

pub(crate) fn same_device<B: Backend + Topology>(
    expected: &ComputeDevice<B>,
    got: &ComputeDevice<B>,
    what: &'static str,
) -> Result<(), Error> {
    if std::ptr::eq(expected, got) {
        Ok(())
    } else {
        Err(Error::OtherDevice { what })
    }
}

pub(crate) fn same_kind(expected: DataKind, got: DataKind) -> Result<(), Error> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::KindMismatch { expected, got })
    }
}

pub(crate) fn same_len(expected: usize, got: usize) -> Result<(), Error> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::LenMismatch { expected, got })
    }
}

pub(crate) fn slot_count(shader: &Signature, got: usize) -> Result<(), Error> {
    if shader.layout.len() == got {
        Ok(())
    } else {
        Err(Error::SlotCount {
            shader: shader.name,
            expected: shader.layout.len(),
            got,
        })
    }
}

// One tensor per slot, each of its slot's kind.
pub(crate) fn slot_kinds(
    shader: &Signature,
    kinds: impl ExactSizeIterator<Item = DataKind>,
) -> Result<(), Error> {
    slot_count(shader, kinds.len())?;
    for (slot, (role, got)) in shader.layout.iter().zip(kinds).enumerate() {
        if role.kind() != got {
            return Err(Error::SlotKind {
                shader: shader.name,
                slot,
                expected: role.kind(),
                got,
            });
        }
    }
    Ok(())
}

// One value per meta field, each of its field's type.
pub(crate) fn meta_types(
    shader: &Signature,
    types: impl ExactSizeIterator<Item = MetaType>,
) -> Result<(), Error> {
    if shader.meta.len() != types.len() {
        return Err(Error::MetaCount {
            shader: shader.name,
            expected: shader.meta.len(),
            got: types.len(),
        });
    }
    for (field, got) in shader.meta.iter().zip(types) {
        if field.ty != got {
            return Err(Error::MetaType {
                shader: shader.name,
                field: field.name,
                expected: field.ty,
                got,
            });
        }
    }
    Ok(())
}

pub(crate) fn accepted<B: Backend>(shader: &Shader) -> Result<(), Error> {
    if B::accepts(shader.binary()) {
        Ok(())
    } else {
        Err(Error::UnsupportedBinary {
            shader: shader.signature().name,
        })
    }
}

pub(crate) fn has_workgroups(shader: &Signature, workgroups: Workgroups) -> Result<(), Error> {
    if workgroups.dims().contains(&0) {
        Err(Error::NoWorkgroups {
            shader: shader.name,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn supported_kind<B: Backend + Topology>(
    dev: &ComputeDevice<B>,
    kind: DataKind,
) -> Result<(), Error> {
    let info = dev.raw.info();
    if info.supports_kind(kind) {
        Ok(())
    } else {
        Err(Error::UnsupportedKind {
            device: info.label(),
            kind,
        })
    }
}

// `Arg::Read` for an `Input` slot, `Arg::Write` for the others.
pub(crate) fn slot_access(
    shader: &Signature,
    slot: usize,
    role: &BindingRole,
    write: bool,
) -> Result<(), Error> {
    if write != !matches!(role, BindingRole::Input(_)) {
        return Err(Error::SlotAccess {
            shader: shader.name,
            slot,
            role: role.clone(),
        });
    }
    Ok(())
}

pub(crate) fn long_enough(
    shader: &Signature,
    slot: usize,
    needs: usize,
    holds: usize,
) -> Result<(), Error> {
    if holds >= needs {
        Ok(())
    } else {
        Err(Error::TooShort {
            shader: shader.name,
            slot,
            needs,
            holds,
        })
    }
}

// A written slot's tensor, named by `tensors` in slot order, is in no other slot.
pub(crate) fn not_aliased<T: PartialEq>(shader: &Signature, tensors: &[T]) -> Result<(), Error> {
    for (slot, (role, written)) in shader.layout.iter().zip(tensors).enumerate() {
        if matches!(role, BindingRole::Input(_)) {
            continue;
        }
        let aliased = tensors
            .iter()
            .enumerate()
            .any(|(other, tensor)| other != slot && tensor == written);
        if aliased {
            return Err(Error::Aliased {
                shader: shader.name,
                slot,
            });
        }
    }
    Ok(())
}
