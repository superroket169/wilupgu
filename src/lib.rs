pub mod backend;
pub mod backends;
pub mod builtin;
pub mod graph;
pub mod id;
pub(crate) mod io_log;
pub mod mesh;
pub mod placement;
pub(crate) mod pool;
pub mod resolver;
pub mod shader;
pub mod tensor;
pub mod topology;
pub mod traits;

pub use backend::{Backend, Binding, Dtype, TensorMode};
#[cfg(feature = "cpu")]
pub use backends::CpuBackend;
#[cfg(feature = "cuda")]
pub use backends::CudaBackend;
pub use backends::WgpuBackend;
pub use graph::{fuse_compute_graphs, ComputeGraph};
pub use shader::{CpuBinding, CudaShape, CudaSpec, MetaField, Shader};
pub use tensor::Tensor;

pub type Real = f32;
