use crate::backend::{Backend, Dtype};
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
        let mut v: Vec<T> = self.ctx.copy_to_cpu(&self.buffer);
        v.truncate(self.size as usize / std::mem::size_of::<T>());
        v
    }

    pub fn new_dtype(ctx: Arc<B>, elem_count: usize, dtype: Dtype) -> Self {
        let buffer = ctx.alloc_dtype(elem_count, dtype);
        let size = (elem_count * dtype.elem_size()) as u64;
        Self { ctx, buffer, size }
    }

    pub fn init_from_cpu_dtype(ctx: Arc<B>, data: &[f32], dtype: Dtype) -> Self {
        let buffer = ctx.alloc_dtype(data.len(), dtype);
        ctx.upload_as(&buffer, data, dtype);
        let size = (data.len() * dtype.elem_size()) as u64;
        Self { ctx, buffer, size }
    }

    pub fn to_cpu_as(&self, dtype: Dtype) -> Vec<f32> {
        let mut v = self.ctx.download_as(&self.buffer, dtype);
        v.truncate(self.size as usize / dtype.elem_size());
        v
    }
}

impl<B: Backend> Drop for Tensor<B> {
    fn drop(&mut self) {
        if B::is_sole_owner(&self.buffer) {
            self.ctx.recycle(self.size, self.buffer.clone());
        }
    }
}
