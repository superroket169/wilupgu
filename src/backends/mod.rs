//! In-tree backend implementations, one file each.

// Always built for tests: the contract's tests run on it.
#[cfg(any(feature = "cpu", test))]
mod cpu;
#[cfg(any(feature = "cpu", test))]
pub use cpu::{CpuArea, CpuBackend, CpuBuffer, CpuCapture, CpuInfo, CpuNode};
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{CudaArea, CudaBackend, CudaBuffer, CudaCapture, CudaInfo, CudaNode};
