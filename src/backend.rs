use crate::shader::Shader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMode {
    Input,
    Output,
    InOut,
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
    fn alloc(&self, size_bytes: u64) -> Self::Buffer;
    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> Self::Buffer;
    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &Self::Buffer, data: &[T]);
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
