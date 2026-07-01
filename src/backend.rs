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

pub fn kernel_layout(name: &str) -> &'static [TensorMode] {
    use TensorMode::*;
    match name {
        // =========== Forward ===========
        "MatMul" => &[Input, Input, Output, Meta],
        "Embedding" => &[Input, Input, Output, Meta],
        "CausalMask" => &[InOut, Meta],
        "SiLU" => &[InOut],
        "RoPE" => &[InOut, Meta],
        "Softmax" => &[InOut, Meta],
        "RMSNorm" => &[Input, Input, Output, Meta],
        "ResidualAdd" => &[InOut, Input],
        "CrossEntropy" => &[Input, Input, Output, Output, Meta],
        "HeadGather" => &[Input, Output, Meta],
        "HeadScatter" => &[Input, Output, Meta],
        "ZeroTensor" => &[Output, Meta],
        "SoftmaxRect" => &[InOut, Meta],
        "CacheWrite" => &[Input, InOut, Meta],
        "RoPEOffset" => &[InOut, Meta],

        // ========= Backward =========
        "MatMulTrp" => &[Input, Input, Output, Meta],
        "MatMulWeightBwd" => &[Input, Input, Output, Meta],
        "SiLUBwd" => &[Input, Input, Output],
        "RoPEBwd" => &[InOut, Meta],
        "SoftmaxBwd" => &[Input, Input, Output, Meta],
        "RMSNormBwd" => &[Input, Input, Input, Output, Output, Meta],
        "RMSNormWeightBwd" => &[Input, Input, Input, Output, Meta],
        "EmbeddingBwd" => &[Input, Input, Output, Meta],
        "CrossEntropyBwd" => &[Input, Input, Input, Output, Meta],
        "BwdAddInplace" => &[InOut, Input],

        // ============ Optimizer ===========
        "AdamW" => &[InOut, Input, InOut, InOut, Meta, Meta],

        _ => panic!("[backend] unknown kernel: {name}"),
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
    fn recycle(&self, size_bytes: u64, buf: Self::Buffer);
    fn build_node(
        &self,
        kernel: &str,
        bindings: &[Binding<Self::Buffer>],
        workgroups: [u32; 3],
    ) -> Self::Node;
    fn execute(&self, nodes: &[Self::Node]);
    fn synchronize(&self);
}
