use crate::backend::{kernel_layout, Backend, Binding};
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
        kernel: &str,
        bindings: &[Binding<B::Buffer>],
        workgroups: [u32; 3],
    ) {
        let layout = kernel_layout(kernel);
        for b in bindings {
            let expected = layout.get(b.slot as usize).unwrap_or_else(|| {
                panic!(
                    "Tensor Mode Mismatch: kernel `{kernel}` binding slot {} out of range (kernel expects {} bindings)",
                    b.slot,
                    layout.len()
                )
            });
            assert_eq!(
                *expected, b.mode,
                "Tensor Mode Mismatch: kernel `{kernel}` slot {} expects {:?}, got {:?}",
                b.slot, expected, b.mode
            );
        }

        let node = self.ctx.build_node(kernel, bindings, workgroups);
        self.nodes.push(node);
    }

    pub fn execute(&self) {
        self.ctx.execute(&self.nodes);
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
