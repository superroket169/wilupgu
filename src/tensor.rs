use crate::backend::Backend;
use std::sync::Arc;

pub struct Tensor<B: Backend> {
    pub ctx: Arc<B>,
    pub buffer: B::Buffer,
    pub size: u64,
}

impl<B: Backend> Tensor<B> {
    pub fn new(ctx: Arc<B>, size_bytes: u64) -> Self {
        let buffer = ctx.alloc(size_bytes);
        Self {
            ctx,
            buffer,
            size: size_bytes,
        }
    }

    pub fn init_from_cpu<T: bytemuck::Pod>(ctx: Arc<B>, data: &[T]) -> Self {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = ctx.alloc_from_cpu(data);
        Self { ctx, buffer, size }
    }

    pub fn copy_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) {
        self.ctx.copy_from_cpu(&self.buffer, data);
    }

    pub fn to_cpu<T: bytemuck::Pod + Default + Clone>(&self) -> Vec<T> {
        self.ctx.copy_to_cpu(&self.buffer)
    }
}

impl<B: Backend> Drop for Tensor<B> {
    fn drop(&mut self) {
        if B::is_sole_owner(&self.buffer) {
            self.ctx.recycle(self.size, self.buffer.clone());
        }
    }
}
