//! The plan tool: describe tensors and calls before a device exists
//! ([`spec`], [`tensor`], [`node`], [`call`], [`run`]), with values known
//! later ([`deferred`]); build them on a device and run them inside its
//! scope ([`runnable`]). [`builder`] makes a plan a step at a time.

pub mod builder;
pub mod call;
pub mod deferred;
pub mod id;
pub mod node;
pub mod run;
pub mod runnable;
pub mod spec;
pub mod tensor;
