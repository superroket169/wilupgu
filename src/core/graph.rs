//! A built, ready-to-run set of nodes. Checks its rules once at `build`,
//! via `rules.rs`; nothing here re-derives a rule `rules.rs` already owns.

use crate::backend::Backend;
use crate::core::node::NodeSpec;
use crate::core::rules::{check_hazards, check_ownership, validate_spec};

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
#[path = "../tests/graph.rs"]
mod tests;
