#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMode {
    Input,
    Output,
    InOut,
    Meta,
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
    fn free_buffer(&self, buf: Self::Buffer);
    fn build_node(
        &self,
        kernel: &str,
        bindings: &[Binding<Self::Buffer>],
        workgroups: [u32; 3],
    ) -> Self::Node;
    fn execute(&self, nodes: &[Self::Node]);
    fn synchronize(&self);
}
