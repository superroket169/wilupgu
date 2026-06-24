use super::{ComputeGraph, ComputeNode};
use crate::context::WgpuContext;
use std::sync::Arc;

pub fn fuse_compute_graphs(ctx: Arc<WgpuContext>, graphs: &[&ComputeGraph]) -> ComputeGraph {
    let mut fused = ComputeGraph::new(ctx);

    for graph in graphs {
        for node in &graph.nodes {
            fused.nodes.push(ComputeNode {
                name: node.name.clone(),
                pipeline: node.pipeline.clone(),
                bind_group: node.bind_group.clone(),
                workgroups: node.workgroups,
            });
        }
    }

    fused
}
