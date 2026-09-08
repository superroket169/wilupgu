pub mod wgpu;
pub use wgpu::WgpuBackend;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
mod cuda_launch_macros;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;

#[cfg(feature = "rayon")]
pub mod rayon;
#[cfg(feature = "rayon")]
pub use rayon::RayonBackend;

#[cfg(feature = "vulkano")]
pub mod vulkano;
