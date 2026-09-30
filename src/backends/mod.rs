pub type CpuBuffer = std::sync::Arc<std::sync::Mutex<Vec<u8>>>;

#[derive(Clone)]
pub struct CpuBinding {
    pub slot: u32,
    pub buffer: CpuBuffer,
}

/// A shader's code, keyed by code format rather than by backend
///
/// wgpu and vulkano both take `Wgsl`
/// cpu and rayon both take `Native`.
pub enum ShaderCode {
    Wgsl(&'static str),
    #[cfg(any(feature = "cpu", feature = "rayon"))]
    Native(fn(&[CpuBinding])),
    // TODO: Cuda(...) - needs a real dispatch type once the CUDA backend is written
}
