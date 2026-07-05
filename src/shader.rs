use crate::backend::TensorMode;
use std::sync::{Arc, Mutex};

pub type CpuBuffer = Arc<Mutex<Vec<u8>>>;

#[derive(Clone)]
pub struct CpuBinding {
    pub slot: u32,
    pub buffer: CpuBuffer,
}

pub struct Shader {
    pub name: &'static str,
    pub layout: &'static [TensorMode],
    pub wgpu: Option<&'static str>,
    pub cpu: Option<fn(&[CpuBinding])>,
    pub cuda: Option<CudaSpec>,
}

pub struct CudaSpec {
    pub src: &'static str,
    pub entry: &'static str,
    pub shape: CudaShape,
}

pub enum CudaShape {
    InOut1,
    In2Out1,
    Add,
    #[cfg(feature = "cuda")]
    Custom(
        fn(
            &'static Shader,
            &crate::backends::cuda::CudaBackend,
            &[crate::backends::cuda::CudaBinding],
            [u32; 3],
        ),
    ),
}
