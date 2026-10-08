//! Backend-independent GPU compute for Rust, aimed at low-end hardware.
//!
//! - [`backend`]: the contract. Backend authors implement its traits; the
//!   raw methods that can cause undefined behavior are `unsafe fn`.
//! - [`backends`]: in-tree backends, each behind its feature (`cpu`).
//! - [`tools`]: optional packages built on the contract, each behind its
//!   feature (`tool-builtins`, on by default).
//!
//! The safe user layer over the raw contract isn't written yet; for now the
//! raw traits are the whole API.

pub mod backend;
pub mod backends;
pub mod tools;
