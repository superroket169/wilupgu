use crate::backend::{Backend, Binding};
use crate::shader::Shader;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static NEXT_GRAPH_ID: AtomicUsize = AtomicUsize::new(0);

pub struct ComputeGraph<B: Backend> {
    ctx: Arc<B>,
    id: usize,
    nodes: Vec<B::Node>,
}

impl<B: Backend> ComputeGraph<B> {
    pub fn new(ctx: Arc<B>) -> Self {
        Self {
            ctx,
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
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
        self.ctx.execute_captured(self.id, &self.nodes);
    }
}

impl<B: Backend> Drop for ComputeGraph<B> {
    fn drop(&mut self) {
        self.ctx.release_captured(self.id);
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
    ComputeGraph {
        ctx,
        id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
        nodes,
    }
}
