//! Optional packages built on the [`backend`](crate::backend) contract and the
//! [`safe`](crate::safe) layer.
//! Each one is behind its own feature; a user can pick a backend and use none of them.

#[cfg(feature = "tool-builtins")]
pub mod builtins;
// TODO: core and spread are out of the build until they're rewritten on the safe layer.
#[cfg(any())]
pub mod core;
#[cfg(any())]
pub mod spread;
