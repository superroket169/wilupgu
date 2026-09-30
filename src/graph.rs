//! A built, ready-to-run set of nodes. Checks its rules once at `build`,
//! via `rules.rs`; nothing here re-derives a rule `rules.rs` already owns.

use crate::backend::Backend;
use crate::node::NodeSpec;
use crate::rules::{check_hazards, check_ownership, validate_spec};

pub enum DispatchPlan {
    Captured {
        key: usize,
        nodes: std::ops::Range<usize>,
    },
    Streamed {
        nodes: std::ops::Range<usize>,
    },
}

pub struct Graph<B: Backend> {
    ctx: std::sync::Arc<B>,
    nodes: Vec<B::Node>,
    plan: Vec<DispatchPlan>,
}

impl<B: Backend> Graph<B> {
    pub fn build(ctx: std::sync::Arc<B>, specs: &[NodeSpec]) -> Result<Self, String> {
        for spec in specs {
            validate_spec::<B::Node>(spec)?;
        }
        check_ownership(ctx.as_ref(), specs)?;
        check_hazards(specs)?;

        let nodes: Vec<B::Node> = specs
            .iter()
            .map(|s| ctx.build_node(s.shader(), s.bindings(), s.workgroups()))
            .collect();
        let plan = vec![DispatchPlan::Streamed {
            nodes: 0..nodes.len(),
        }];

        Ok(Self { ctx, nodes, plan })
    }

    pub fn run(&self) {
        for segment in &self.plan {
            match segment {
                DispatchPlan::Streamed { nodes } => self.ctx.execute(&self.nodes[nodes.clone()]),
                DispatchPlan::Captured { key, nodes } => {
                    self.ctx.execute_captured(*key, &self.nodes[nodes.clone()])
                }
            }
        }
    }

    pub fn capture(&mut self, key: usize, range: std::ops::Range<usize>) {
        let mut new_plan = Vec::with_capacity(self.plan.len() + 2);
        for segment in std::mem::take(&mut self.plan) {
            match segment {
                DispatchPlan::Streamed { nodes } => {
                    let overlap_start = nodes.start.max(range.start);
                    let overlap_end = nodes.end.min(range.end);
                    if overlap_start >= overlap_end {
                        new_plan.push(DispatchPlan::Streamed { nodes });
                        continue;
                    }
                    if nodes.start < overlap_start {
                        new_plan.push(DispatchPlan::Streamed {
                            nodes: nodes.start..overlap_start,
                        });
                    }
                    new_plan.push(DispatchPlan::Captured {
                        key,
                        nodes: overlap_start..overlap_end,
                    });
                    if overlap_end < nodes.end {
                        new_plan.push(DispatchPlan::Streamed {
                            nodes: overlap_end..nodes.end,
                        });
                    }
                }
                already_captured => new_plan.push(already_captured),
            }
        }
        self.plan = new_plan;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::tests::{copy_node, device_with};

    #[test]
    fn graph_build_and_run() {
        let (ctx, ids) = device_with(2);
        let graph = Graph::build(ctx, &[copy_node(ids[0], ids[1])]).unwrap();
        graph.run();
    }

    #[test]
    fn graph_capture_falls_back_to_execute() {
        let (ctx, ids) = device_with(2);
        let mut graph = Graph::build(ctx, &[copy_node(ids[0], ids[1])]).unwrap();
        graph.capture(7, 0..1);
        assert!(matches!(
            graph.plan.as_slice(),
            [DispatchPlan::Captured { key: 7, .. }]
        ));
        graph.run(); // ToyBackend has no real capture support, runs via the default fallback
    }

    #[test]
    fn graph_build_rejects_unordered_double_write() {
        let (ctx, ids) = device_with(2);
        let specs = [copy_node(ids[0], ids[1]), copy_node(ids[0], ids[1])];
        let err = Graph::build(ctx, &specs).err().unwrap();
        assert!(err.contains("Buffer hazard"), "unexpected error: {err}");
    }

    #[test]
    fn graph_build_rejects_foreign_buffer() {
        let (ctx, ids) = device_with(1);
        let (_other, foreign) = device_with(1); // allocated on a different device
        let err = Graph::build(ctx, &[copy_node(ids[0], foreign[0])])
            .err()
            .unwrap();
        assert!(
            err.contains("Buffer ownership mismatch"),
            "unexpected error: {err}"
        );
    }
}
