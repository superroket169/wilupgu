use crate::context::WgpuContext;
use std::sync::Arc;

pub struct Tensor {
    pub ctx: Arc<WgpuContext>,
    pub buffer: Arc<wgpu::Buffer>,
    pub size: wgpu::BufferAddress,
    // TODO: not surely. can add Vec<usize>
}

impl Tensor {
    pub fn new(ctx: Arc<WgpuContext>, size_bytes: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wilupgu_Tensor"),
            size: size_bytes,
            usage,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            buffer: Arc::new(buffer),
            size: size_bytes,
        }
    }
}
