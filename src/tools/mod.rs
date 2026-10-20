//! Optional packages built on the [`backend`](crate::backend) contract and the
//! [`safe`](crate::safe) layer.
//! Each one is behind its own feature; a user can pick a backend and use none of them.

#[cfg(feature = "tool-builtins")]
pub mod builtins;
// Always built for tests, like the CPU backend.
#[cfg(any(feature = "tool-core", test))]
pub mod core;
// TODO: spread is out of the build until it's rewritten on the safe layer.
#[cfg(any())]
pub mod spread;
