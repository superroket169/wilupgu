//! The plan tool: describe tensors and calls before a device exists
//! ([`tensor`], [`node`], [`call`], [`run`]), with values known
//! later ([`deferred`]); build them on a device and run them inside its
//! scope ([`run`]). [`mesh`] holds routines of calls, each run by its key.
//! [`builder`] makes a plan a step at a time.

pub mod builder;
pub mod call;
pub mod deferred;
pub mod id;
pub mod mesh;
pub mod node;
pub mod run;
pub mod tensor;
