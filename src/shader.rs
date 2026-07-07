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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaField {
    U32,
    F32,
}

pub enum CudaShape {
    Generic {
        meta_fields: &'static [MetaField],
        block_dim: (u32, u32, u32),
    },

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
