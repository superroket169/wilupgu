use crate::context::WgpuContext;
use crate::tensor::Tensor;
use std::sync::Arc;

pub struct ComputeNode {
    pub name: String,
    pub shader_module: wgpu::ShaderModule,
    // TODO: tensors will be knowing just "reading" or "writing" its important
    pub inputs: Vec<Arc<Tensor>>,
    pub outputs: Vec<Arc<Tensor>>,
    pub workgroups: [u32; 3],
}

pub struct ComputeGraph {
    ctx: Arc<WgpuContext>,
    nodes: Vec<ComputeNode>,
}

impl ComputeGraph {
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        Self {
            ctx,
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: ComputeNode) {
        self.nodes.push(node);
    }

    pub fn compile_and_execute(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Wilupgu_Graph_Encoder"),
            });

        // TODO: Topo-Sort and ComputePass will be implement here

        self.ctx.queue.submit(Some(encoder.finish()));
    }
}
