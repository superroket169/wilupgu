//! Borrow-checked GPU dispatch for Rust, with optional tools on top.
//!
//! GPU work runs in a scope, like `std::thread::scope`: a buffer bound to it
//! stays borrowed until the device is done with it, and every run is checked
//! against its shader before it reaches the device. The same code runs on
//! CUDA, wgpu and a CPU reference backend.
//!
//! ```
//! use wilupgu::backends::CpuBackend;
//! use wilupgu::prelude::*;
//! use wilupgu::tools::builtins::ADD;
//!
//! # fn main() -> Result<(), Error> {
//! let dev = ComputeDevice::<CpuBackend>::first()?;
//! let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0]))?;
//! let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 20.0, 30.0]))?;
//!
//! // x[i] += y[i] for i < n; the meta is n.
//! let n = 3;
//! let shader = ADD.shader_for::<CpuBackend>()?;
//! let wg = Workgroups::cover(n, &shader);
//! let mut add = dev.node(&shader, &[MetaValue::Uint(n)], wg)?;
//!
//! // While bound, the host can't touch x or y; the scope waits for the
//! // device before it returns.
//! dev.scope(|s| {
//!     let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
//!     s.run(add, &[Arg::Write(x), Arg::Read(y)])
//! })?;
//! assert_eq!(x.download_vec::<F32>()?, vec![11.0, 22.0, 33.0]);
//! # Ok(())
//! # }
//! ```
//!
//! Reading `x` while the scope still holds it doesn't compile:
//!
//! ```compile_fail
//! # use wilupgu::backends::CpuBackend;
//! # use wilupgu::prelude::*;
//! # use wilupgu::tools::builtins::ADD;
//! # fn main() -> Result<(), Error> {
//! # let dev = ComputeDevice::<CpuBackend>::first()?;
//! # let mut x = dev.tensor_from(&F32::wrap(vec![1.0]))?;
//! # let mut y = dev.tensor_from(&F32::wrap(vec![10.0]))?;
//! # let shader = ADD.shader_for::<CpuBackend>()?;
//! # let mut add = dev.node(&shader, &[MetaValue::Uint(1)], Workgroups::cover(1, &shader))?;
//! dev.scope(|s| {
//!     let (xs, ys, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
//!     s.run(add, &[Arg::Write(xs), Arg::Read(ys)])?;
//!     x.download_vec::<F32>()?; // error[E0502]: `x` is still borrowed
//!     Ok(())
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! - [`safe`]: the API users call: devices, buffers, nodes, scopes and
//!   recordings, with every check written once for all backends.
//! - [`backend`]: the contract. Backend authors implement its traits; the
//!   raw methods that can cause undefined behavior are `unsafe fn`, and
//!   [`safe`] is their only caller.
//! - [`backends`]: in-tree backends, each behind its feature (`cpu`,
//!   `cuda`, `wgpu`).
//! - [`tools`]: optional packages built on top, each behind its feature
//!   (`tool-builtins`, on by default).

pub mod backend;
pub mod backends;
mod error;
pub mod prelude;
pub mod safe;
pub mod tools;

pub use error::Error;
