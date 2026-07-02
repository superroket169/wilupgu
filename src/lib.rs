pub mod backend;
pub mod backends;
pub mod graph;
pub mod nn;
pub(crate) mod pool;
pub mod tensor;

pub use backend::{Backend, Binding, TensorMode};
pub use backends::WgpuBackend;
#[cfg(feature = "cuda")]
pub use backends::CudaBackend;
#[cfg(feature = "cpu")]
pub use backends::CpuBackend;
pub use graph::{fuse_compute_graphs, ComputeGraph};
pub use tensor::Tensor;

pub type Real = f32;
