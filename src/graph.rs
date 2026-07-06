use crate::backend::{Backend, Binding};
use crate::shader::Shader;
use std::sync::Arc;

pub struct ComputeGraph<B: Backend> {
    ctx: Arc<B>,
    nodes: Vec<B::Node>,
}

impl<B: Backend> ComputeGraph<B> {
    pub fn new(ctx: Arc<B>) -> Self {
        Self {
            ctx,
            nodes: Vec::new(),
        }
    }

    pub fn add_node(
        &mut self,
        shader: &'static Shader,
        bindings: &[Binding<B::Buffer>],
        workgroups: [u32; 3],
    ) {
        let layout = shader.layout;
        let name = shader.name;
        for b in bindings {
            let expected = layout.get(b.slot as usize).unwrap_or_else(|| {
                panic!(
                    "Tensor Mode Mismatch: kernel `{name}` binding slot {} out of range (kernel expects {} bindings)",
                    b.slot,
                    layout.len()
                )
            });
            assert_eq!(
                *expected, b.mode,
                "Tensor Mode Mismatch: kernel `{name}` slot {} expects {:?}, got {:?}",
                b.slot, expected, b.mode
            );
        }

        let node = self.ctx.build_node(shader, bindings, workgroups);
        self.nodes.push(node);
    }

    pub fn execute(&self) {
        self.ctx.execute(&self.nodes);
    }

    pub fn execute_captured(&self) {
        let key = self.nodes.as_ptr() as usize;
        self.ctx.execute_captured(key, &self.nodes);
    }
}

pub fn fuse_compute_graphs<B: Backend>(
    ctx: Arc<B>,
    graphs: &[&ComputeGraph<B>],
) -> ComputeGraph<B> {
    let nodes = graphs
        .iter()
        .flat_map(|g| g.nodes.iter().cloned())
        .collect();
    ComputeGraph { ctx, nodes }
}
