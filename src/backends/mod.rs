//! Concrete backend implementations live here, one file each, once written.

// Always built for tests: the contract's tests run on it.
#[cfg(any(feature = "cpu", test))]
mod cpu;
#[cfg(any(feature = "cpu", test))]
pub use cpu::{CpuBackend, CpuBuffer, CpuInfo, CpuNode};
