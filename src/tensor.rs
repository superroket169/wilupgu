use crate::context::WgpuContext;
use bytemuck::Pod;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct Tensor {
    pub ctx: Arc<WgpuContext>,
    pub buffer: Arc<wgpu::Buffer>,
    pub size: wgpu::BufferAddress,
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

    pub fn init_from_cpu<T: Pod>(ctx: Arc<WgpuContext>, data: &[T]) -> Self {
        let size = (data.len() * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Wilupgu_Tensor"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            ctx,
            buffer: Arc::new(buffer),
            size,
        }
    }

    pub fn copy_from_cpu<T: Pod>(&self, data: &[T]) {
        self.ctx
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn to_cpu<T: Pod + Default + Clone>(&self) -> Vec<T> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging_Buffer"),
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(async { receiver.receive().await.unwrap().unwrap() });

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        result
    }

    pub fn free(self) {
        self.buffer.destroy();
    }
}
