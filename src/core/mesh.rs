use crate::core::node::NodeId;
use crate::core::run::RunPlan;

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
