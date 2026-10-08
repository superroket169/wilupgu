//! Concrete backend implementations live here, one file each, once written.

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::{CpuBackend, CpuBuffer, CpuInfo, CpuNode};

#[cfg(test)]
pub(crate) mod toy;
