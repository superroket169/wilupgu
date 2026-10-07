//! A built, ready-to-run set of nodes. Checks its rules once at `build`,
//! via `rules.rs`; nothing here re-derives a rule `rules.rs` already owns.

use std::collections::HashMap;

use crate::backend::Backend;
use crate::tools::core::node::{MetaValue, NodeSpec};
use crate::tools::core::rules::{
    check_hazards, check_meta, check_ownership, check_shader_code, validate_spec,
};

pub enum DispatchPlan {
    Captured {
        key: usize,
        nodes: std::ops::Range<usize>,
    },
    Streamed {
        nodes: std::ops::Range<usize>,
    },
}

pub struct Graph<'d, B: Backend> {
    ctx: &'d B,
    nodes: Vec<B::Node>,
    // Each node's buffers, held here so they live as long as the node can run,
    // whatever the backend's node keeps. Only held, never read.
    #[allow(dead_code)]
    buffers: Vec<Vec<B::Buffer>>,
    // Nodes with per-run meta fields: index into `nodes`, and all of that node's meta.
    per_run_meta: Vec<(usize, Vec<MetaValue>)>,
    plan: Vec<DispatchPlan>,
}

impl<'d, B: Backend> Graph<'d, B> {
    pub fn build(ctx: &'d B, specs: &[NodeSpec]) -> Result<Self, String> {
        for spec in specs {
            validate_spec::<B::Node>(spec)?;
            check_meta(spec)?;
        }
        check_shader_code::<B>(specs)?;
        check_ownership(ctx, specs)?;
        check_hazards(specs)?;

        let mut nodes: Vec<B::Node> = Vec::with_capacity(specs.len());
        let mut buffers: Vec<Vec<B::Buffer>> = Vec::with_capacity(specs.len());
        for s in specs {
            // Once values are read here; per-run ones get a placeholder until `run`.
            let words: Vec<u32> = s
                .meta()
                .iter()
                .map(|v| v.once_word().unwrap_or(0))
                .collect();
            let bound: Vec<_> = s
                .bindings()
                .iter()
                .map(|b| {
                    let buf = ctx
                        .table()
                        .get(b.tensor)
                        .expect("checked by check_ownership");
                    (b.slot, b.mode.clone(), buf)
                })
                .collect();
            let node = ctx.build_node(s.shader(), &words, &bound, s.workgroups());
            nodes.push(node);
            buffers.push(bound.into_iter().map(|(_, _, buf)| buf).collect());
        }
        let per_run_meta = specs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.meta().iter().any(MetaValue::is_per_run))
            .map(|(i, s)| (i, s.meta().to_vec()))
            .collect();
        let plan = vec![DispatchPlan::Streamed {
            nodes: 0..nodes.len(),
        }];

        Ok(Self {
            ctx,
            nodes,
            buffers,
            per_run_meta,
            plan,
        })
    }

    pub fn run(&self) {
        self.write_per_run_meta();
        for segment in &self.plan {
            match segment {
                DispatchPlan::Streamed { nodes } => self.ctx.execute(&self.nodes[nodes.clone()]),
                DispatchPlan::Captured { key, nodes } => {
                    self.ctx.execute_captured(*key, &self.nodes[nodes.clone()])
                }
            }
        }
    }

    // Each `Dynamic` is read once per run, however many nodes share it.
    fn write_per_run_meta(&self) {
        let mut read: HashMap<usize, u32> = HashMap::new();
        for (node, meta) in &self.per_run_meta {
            let words: Vec<u32> = meta
                .iter()
                .map(|v| match v.dynamic_key() {
                    Some(key) => *read.entry(key).or_insert_with(|| v.read_word_for_run()),
                    None => v.read_word_for_run(),
                })
                .collect();
            self.ctx.update_meta(&self.nodes[*node], &words);
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
#[path = "../../tests/graph.rs"]
mod tests;
