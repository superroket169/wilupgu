//! The [`Spec`] trait: a description of something to build on a device,
//! checked when it's made and built later, once every value it waits on is
//! known.

use crate::backend::{Backend, Topology};
use crate::safe::ComputeDevice;
use crate::tools::plan::id::GlobalId;

/// What will exist on a device: made from a plain description, built on any
/// device of any backend once its [`Resolvable`](super::deferred::Resolvable)s
/// are known. A spec doesn't change after it's made.
pub trait Spec: Sized {
    /// What the user fills in to make the spec.
    type Description;
    /// What [`Spec::build`] makes on one device.
    type Outcome<'d, B: Backend + Topology + 'd>;

    /// Checks everything the description can tell on its own, before any
    /// device or later value exists.
    fn new(description: Self::Description) -> Result<Self, String>;

    /// This spec's id; nothing else carries it.
    // Only `TensorSpec` and `CallSpec` ids are looked up for now; `NodeSpec`
    // and `RunPlan` keep theirs for the tools to come (autograd, mesh).
    fn id(&self) -> GlobalId<Self>;

    /// Fails, naming what is missing, while a value the build needs isn't
    /// known yet, or once known, doesn't fit the description.
    fn ready(&self) -> Result<(), String>;

    /// Builds the spec on `device`. Fails where [`Spec::ready`] does, or
    /// where the device does.
    fn build<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Self::Outcome<'d, B>, String>;
}
