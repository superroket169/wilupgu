//! Backend-independent GPU compute for Rust, aimed at low-end hardware.
//!
//! - [`safe`]: the API users call: devices, buffers, nodes, scopes and
//!   recordings, with every check written once for all backends.
//! - [`backend`]: the contract. Backend authors implement its traits; the
//!   raw methods that can cause undefined behavior are `unsafe fn`, and
//!   [`safe`] is their only caller.
//! - [`backends`]: in-tree backends, each behind its feature (`cpu`).
//! - [`tools`]: optional packages built on top, each behind its feature
//!   (`tool-builtins`, on by default).

pub mod backend;
pub mod backends;
pub mod safe;
pub mod tools;
