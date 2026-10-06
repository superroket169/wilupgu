use crate::core::node::NodeId;

/// One partition of a compiled graph that runs as a single submit.
/// Where the boundaries fall depends on the run strategy
///
/// # example:
/// CUDA capture: the whole mesh
/// streaming: one node
///
/// a split/combine boundary always ends one.
pub struct RunPart {
    nodes: Vec<NodeId>,
}

impl RunPart {
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}

pub struct RunPlan {
    parts: Vec<RunPart>,
}

impl RunPlan {
    pub fn parts(&self) -> &[RunPart] {
        &self.parts
    }
}
