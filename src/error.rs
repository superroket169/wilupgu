//! [`Error`]: every error the safe layer and the tools return.

use std::fmt;

use crate::backend::dtype::DataKind;
use crate::backend::shader::{BindingRole, MetaType};

/// Why a call failed. Most variants are a broken rule, found before
/// anything reaches the device; [`Error::Backend`] is the backend's own.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A tensor, node or recording of another device.
    OtherDevice { what: &'static str },
    /// Elements of another kind.
    KindMismatch { expected: DataKind, got: DataKind },
    /// Another count of elements.
    LenMismatch { expected: usize, got: usize },
    /// Another count of tensors than the shader has slots.
    SlotCount {
        shader: &'static str,
        expected: usize,
        got: usize,
    },
    /// A tensor of another kind than its slot holds.
    SlotKind {
        shader: &'static str,
        slot: usize,
        expected: DataKind,
        got: DataKind,
    },
    /// [`Arg::Read`](crate::safe::Arg::Read) for a written slot, or
    /// [`Arg::Write`](crate::safe::Arg::Write) for an `Input` one.
    SlotAccess {
        shader: &'static str,
        slot: usize,
        role: BindingRole,
    },
    /// A tensor shorter than the shader's `min_lens` for the node's meta.
    TooShort {
        shader: &'static str,
        slot: usize,
        needs: usize,
        holds: usize,
    },
    /// A written slot's tensor in another slot too.
    Aliased { shader: &'static str, slot: usize },
    /// Another count of meta values than the shader has fields.
    MetaCount {
        shader: &'static str,
        expected: usize,
        got: usize,
    },
    /// A meta value of another type than its field.
    MetaType {
        shader: &'static str,
        field: &'static str,
        expected: MetaType,
        got: MetaType,
    },
    /// The shader's binary is of a form the backend doesn't take.
    UnsupportedBinary { shader: &'static str },
    /// No device of this backend on this machine.
    NoDevice { backend: &'static str },
    /// A workgroup count of 0 in some dimension.
    NoWorkgroups { shader: &'static str },
    /// The device doesn't support this kind.
    UnsupportedKind { device: String, kind: DataKind },
    /// Elements `offset..offset + len` don't fit a tensor of `tensor_len`.
    OutOfRange {
        offset: usize,
        len: usize,
        tensor_len: usize,
    },
    /// A range starts, or a write ends, at `element`, which isn't a multiple
    /// of [`COPY_ALIGN`](crate::backend::COPY_ALIGN) bytes (nor, for a
    /// write's end, the tensor's end).
    Misaligned { kind: DataKind, element: usize },
    /// A copy from a tensor into itself.
    SameTensor,
    /// A capture inside a capture.
    AlreadyCapturing,
    /// A replay inside a capture.
    ReplayWhileCapturing,
    /// A replay's recorded node or tensor isn't bound in the scope, or has
    /// moved.
    NotBound { what: &'static str },
    /// An id of a tensor or call that isn't in the plan.
    NotInPlan { what: &'static str },
    /// A [`Resolvable`](crate::tools::plan::deferred::Resolvable) read
    /// before its value was given.
    NotGivenYet,
    /// A resolver asked for twice.
    ResolverTaken,
    /// A resolver asked for a value a `Relation` computes.
    Computed,
    /// A [`Dynamic`](crate::tools::plan::deferred::Dynamic) read before it
    /// was ever set.
    DynamicUnset,
    /// A compiler failed; `log` is its report.
    Compile { compiler: &'static str, log: String },
    /// A run's call at `index` in its order failed with `error`.
    InCall { index: usize, error: Box<Error> },
    /// The backend's own error.
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OtherDevice { what } => write!(f, "the {what} belongs to another device"),
            Error::KindMismatch { expected, got } => write!(f, "expected {expected:?}, got {got:?}"),
            Error::LenMismatch { expected, got } => {
                write!(f, "expected {expected} elements, got {got}")
            }
            Error::SlotCount {
                shader,
                expected,
                got,
            } => write!(f, "shader `{shader}` has {expected} slots, got {got} tensors"),
            Error::SlotKind {
                shader,
                slot,
                expected,
                got,
            } => write!(
                f,
                "shader `{shader}`: slot {slot} holds {expected:?}, its tensor {got:?}"
            ),
            Error::SlotAccess { shader, slot, role } => {
                let arg = match role {
                    BindingRole::Input(_) => "Read",
                    _ => "Write",
                };
                write!(f, "shader `{shader}`: slot {slot} is {role:?}, it needs Arg::{arg}")
            }
            Error::TooShort {
                shader,
                slot,
                needs,
                holds,
            } => write!(
                f,
                "shader `{shader}`: slot {slot} needs {needs} elements for this meta, its tensor holds {holds}"
            ),
            Error::Aliased { shader, slot } => write!(
                f,
                "shader `{shader}`: slot {slot} is written, its tensor can't be in another slot"
            ),
            Error::MetaCount {
                shader,
                expected,
                got,
            } => write!(f, "shader `{shader}` has {expected} meta fields, got {got}"),
            Error::MetaType {
                shader,
                field,
                expected,
                got,
            } => write!(
                f,
                "shader `{shader}`: meta field `{field}` is {expected:?}, got {got:?}"
            ),
            Error::UnsupportedBinary { shader } => {
                write!(f, "shader `{shader}`: the backend doesn't take this binary")
            }
            Error::NoDevice { backend } => write!(f, "{backend}: no device on this machine"),
            Error::NoWorkgroups { shader } => {
                write!(f, "shader `{shader}`: a workgroup count of 0")
            }
            Error::UnsupportedKind { device, kind } => {
                write!(f, "device `{device}` doesn't support {kind:?}")
            }
            Error::OutOfRange {
                offset,
                len,
                tensor_len,
            } => write!(
                f,
                "elements {offset}..{offset} + {len} don't fit a tensor of {tensor_len}"
            ),
            Error::Misaligned { kind, element } => write!(
                f,
                "element {element} of {kind:?} isn't at a multiple of {} bytes",
                crate::backend::COPY_ALIGN
            ),
            Error::SameTensor => write!(f, "a copy from a tensor into itself"),
            Error::AlreadyCapturing => write!(f, "a capture inside a capture"),
            Error::ReplayWhileCapturing => write!(f, "a replay inside a capture"),
            Error::NotBound { what } => write!(
                f,
                "a recorded {what} isn't bound in this scope, or has moved"
            ),
            Error::NotInPlan { what } => write!(f, "this {what} isn't in the plan"),
            Error::NotGivenYet => write!(f, "a value was read before it was given"),
            Error::ResolverTaken => write!(f, "the resolver of this value was already taken"),
            Error::Computed => write!(f, "this value is computed by a Relation"),
            Error::DynamicUnset => write!(f, "a Dynamic was read before it was ever set"),
            Error::Compile { compiler, log } => write!(f, "{compiler}: {log}"),
            Error::InCall { index, error } => write!(f, "call {index}: {error}"),
            Error::Backend(e) => write!(f, "backend: {e}"),
        }
    }
}

impl std::error::Error for Error {}
