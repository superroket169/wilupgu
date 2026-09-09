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

pub type CpuBuffer = std::sync::Arc<std::sync::Mutex<Vec<u8>>>;

#[derive(Clone)]
pub struct CpuBinding {
    pub slot: u32,
    pub buffer: CpuBuffer,
}

pub enum BackendDispatch {
    Wgpu(&'static str),
    #[cfg(feature = "cpu")]
    Cpu(fn(&[CpuBinding])),
    #[cfg(feature = "rayon")]
    Rayon(fn(&[CpuBinding])),
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaDispatch),
}
