//! Optional packages built on the [`backend`](crate::backend) contract.
//! Each one is behind its own feature; a user can pick a backend and use none of them.

#[cfg(feature = "tool-builtins")]
pub mod builtins;
// TODO: core and spread are out of the build until they move to the ownership contract.
#[cfg(any())]
pub mod core;
#[cfg(any())]
pub mod spread;
