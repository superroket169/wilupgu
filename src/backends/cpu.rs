use crate::backend::{Backend, Binding, Dtype};
use crate::pool::BufferPool;
use crate::shader::{CpuBinding, CpuBuffer, Shader};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct CpuNode {
    func: fn(&[CpuBinding]),
    bindings: Vec<CpuBinding>,
}

pub struct CpuBackend {
    pool: BufferPool<CpuBuffer>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            pool: BufferPool::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;
    type Node = CpuNode;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn alloc(&self, size_bytes: u64) -> CpuBuffer {
        if let Some(buf) = self.pool.take(size_bytes) {
            return buf;
        }
        Arc::new(Mutex::new(vec![0u8; size_bytes as usize]))
    }

    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> CpuBuffer {
        let bytes = bytemuck::cast_slice(data);
        if let Some(buf) = self.pool.take(bytes.len() as u64) {
            buf.lock().unwrap().copy_from_slice(bytes);
            return buf;
        }
        Arc::new(Mutex::new(bytes.to_vec()))
    }

    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &CpuBuffer, data: &[T]) {
        buf.lock()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(data));
    }

    fn copy_to_cpu<T: bytemuck::Pod + Default + Clone>(&self, buf: &CpuBuffer) -> Vec<T> {
        let g = buf.lock().unwrap();
        bytemuck::cast_slice::<u8, T>(&g).to_vec()
    }

    fn alloc_dtype(&self, elem_count: usize, dtype: Dtype) -> CpuBuffer {
        assert_eq!(
            dtype,
            Dtype::F32,
            "[cpu] backend only supports F32, got {dtype:?}"
        );
        self.alloc((elem_count * dtype.elem_size()) as u64)
    }

    fn upload_as(&self, buf: &CpuBuffer, data: &[f32], dtype: Dtype) {
        assert_eq!(
            dtype,
            Dtype::F32,
            "[cpu] backend only supports F32, got {dtype:?}"
        );
        self.copy_from_cpu(buf, data);
    }

    fn download_as(&self, buf: &CpuBuffer, dtype: Dtype) -> Vec<f32> {
        assert_eq!(
            dtype,
            Dtype::F32,
            "[cpu] backend only supports F32, got {dtype:?}"
        );
        self.copy_to_cpu(buf)
    }

    fn free_buffer(&self, _buf: CpuBuffer) {}

    fn recycle(&self, size_bytes: u64, buf: CpuBuffer) {
        let _ = self.pool.recycle(size_bytes, buf);
    }

    fn is_sole_owner(buf: &CpuBuffer) -> bool {
        Arc::strong_count(buf) == 1
    }

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<CpuBuffer>],
        _workgroups: [u32; 3],
    ) -> CpuNode {
        let func = shader
            .cpu
            .unwrap_or_else(|| panic!("[cpu] shader `{}` has no cpu impl", shader.name));
        CpuNode {
            func,
            bindings: bindings
                .iter()
                .map(|b| CpuBinding {
                    slot: b.slot,
                    buffer: b.buffer.clone(),
                })
                .collect(),
        }
    }

    fn execute(&self, nodes: &[CpuNode]) {
        for node in nodes {
            (node.func)(&node.bindings);
        }
    }

    fn synchronize(&self) {}
}
