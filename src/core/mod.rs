//! The crate's own domain: dtype/tensor/node/shader/graph definitions and the
//! rules and planning built on them. `backend.rs` (the contract) and
//! `backends/` (the implementations) live one level up, on purpose -- they
//! aren't part of this crate's own subject matter, they're what runs it.

pub mod builtins;
pub mod deferred;
pub mod device;
pub mod dtype;
pub mod graph;
pub mod id;
pub(crate) mod io_log;
pub mod mesh;
pub mod node;
pub mod placement;
pub(crate) mod pool;
pub(crate) mod rules;
pub mod run;
pub mod shader;
pub mod spread;
pub mod table;
pub mod tensor;
