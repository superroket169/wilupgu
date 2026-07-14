use crate::shader::Shader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMode {
    Input,
    /// Fully overwritten by the kernel: prior contents (pool garbage
    /// included) never leak into the result.
    Output,
    /// Read-modify-write where the old value is consumed (AdamW weights,
    /// in-place masking).
    InOut,
    /// Read-add-write: the kernel does `buf += result`, so the caller must
    /// hand over initialized (usually zeroed or partial-sum) contents.
    /// Backends treat it like InOut; the mode exists so layouts state the
    /// init contract instead of hiding it behind Output/InOut.
    Accumulate,
    Meta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    pub fn elem_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }
}

pub struct Binding<'a, Buf> {
    pub slot: u32,
    pub buffer: &'a Buf,
    pub mode: TensorMode,
}

impl<'a, Buf> Binding<'a, Buf> {
    #[inline]
    pub fn new(slot: u32, buffer: &'a Buf, mode: TensorMode) -> Self {
        Self { slot, buffer, mode }
    }
}

pub trait Backend: Send + Sync + 'static {
    type Buffer: Clone + Send + Sync + 'static;
    type Node: Clone + Send + Sync + 'static;
    fn name(&self) -> &'static str;

    /// Returned buffer may come from the pool: contents are GARBAGE, and the
    /// physical size can exceed `size_bytes` (wgpu rounds up to a power-of-two
    /// size class). CUDA additionally asserts `size_bytes % 4 == 0`.
    fn alloc(&self, size_bytes: u64) -> Self::Buffer;

    /// CUDA stores every buffer as f32 words -- `T` must be 4-byte aligned.
    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> Self::Buffer;

    /// Ordered against queued kernel work on the backend's single queue/stream:
    /// safe to call between executes without a synchronize().
    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &Self::Buffer, data: &[T]);

    /// Blocking. Returns the PHYSICAL buffer contents -- on wgpu the tail beyond
    /// the logical tensor size is pool garbage. Go through `Tensor::to_cpu`,
    /// which truncates to the logical size.
    fn copy_to_cpu<T: bytemuck::Pod + Default + Clone>(&self, buf: &Self::Buffer) -> Vec<T>;

    // dynamic type quantization like somethings
    fn alloc_dtype(&self, elem_count: usize, dtype: Dtype) -> Self::Buffer;
    fn upload_as(&self, buf: &Self::Buffer, data: &[f32], dtype: Dtype);
    fn download_as(&self, buf: &Self::Buffer, dtype: Dtype) -> Vec<f32>;

    fn free_buffer(&self, buf: Self::Buffer);
    fn recycle(&self, size_bytes: u64, buf: Self::Buffer);
    fn is_sole_owner(buf: &Self::Buffer) -> bool;
    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<Self::Buffer>],
        workgroups: [u32; 3],
    ) -> Self::Node;
    fn execute(&self, nodes: &[Self::Node]);
    fn execute_captured(&self, _key: usize, nodes: &[Self::Node]) {
        self.execute(nodes);
    }
    fn release_captured(&self, _key: usize) {}
    fn synchronize(&self);
}
