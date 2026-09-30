use crate::node::NodeId;

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

/// Multi-device execution over a compiled graph
///
/// skeleton only
pub struct ComputeMesh {
    plan: RunPlan,
}

impl ComputeMesh {
    /// The plan `build` produced: what parts exist and their run order.
    pub fn plan(&self) -> &RunPlan {
        &self.plan
    }

    /// The node that last finished running, relative to `plan()`.
    pub fn last_run(&self) -> Option<NodeId> {
        todo!()
    }

    /// Runs a single part — stepping through by hand, or driving parts directly.
    pub fn run_part(&mut self, _part: usize) -> Result<(), String> {
        todo!()
    }

    /// Runs every part up to and including the one containing `node`.
    pub fn run_until(&mut self, _node: NodeId) -> Result<(), String> {
        todo!()
    }

    /// Runs every remaining part, start to end. what the main loop calls.
    pub fn run_whole(&mut self) -> Result<(), String> {
        todo!()
    }
}
