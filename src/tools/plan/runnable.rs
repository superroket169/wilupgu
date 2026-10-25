//! [`Runnable`]: a built [`RunPlan`], run inside the caller's scope. On a
//! [`Capturable`] backend it can become a [`CapturedRunnable`], which replays
//! one recording of the whole run.
//!
//! Neither opens a scope to run: [`Runnable::bind`] and
//! [`CapturedRunnable::bind`] bind everything into the caller's scope once,
//! and the handle they return runs as many times as needed in it.

use std::collections::{HashMap, HashSet};

use crate::backend::dtype::HostData;
use crate::backend::shader::BindingRole;
use crate::backend::{Backend, Capturable, Topology};
use crate::safe::{
    Arg, ComputeDevice, MetaValue, Node, NodeSlot, Recording, Scope, Tensor, TensorSlot,
};
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::node::{MetaSource, NodeId};
use crate::tools::plan::run::{role_kind, RunPlan};
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::TensorId;

/// A built [`RunPlan`]: its tensors and nodes on one device, and the order the
/// nodes run in. Everything in it can be read; [`Runnable::edit`] changes it
/// between runs, [`Runnable::bind`] runs it inside a scope.
pub struct Runnable<'d, B: Backend + Topology> {
    device: &'d ComputeDevice<B>,
    tensors: Vec<Tensor<'d, B>>,
    // The spec id of each tensor, in the same order.
    tensor_ids: Vec<TensorId>,
    nodes: Vec<Node<'d, B>>,
    // The spec id of each node, in the same order.
    node_ids: Vec<NodeId>,
    // For each node, the index in `tensors` of each slot's tensor.
    slots: Vec<Vec<usize>>,
    // For each node, where each meta field comes from.
    sources: Vec<Vec<MetaSource>>,
    // The run order, as indexes in `nodes`.
    order: Vec<usize>,
}

fn index_of<T>(ids: &[GlobalId<T>], id: GlobalId<T>, what: &str) -> Result<usize, String> {
    ids.iter()
        .position(|&i| i == id)
        .ok_or_else(|| format!("this {what} isn't in the plan"))
}

impl<'d, B: Backend + Topology> Runnable<'d, B> {
    // `RunPlan::build`, once the plan is ready.
    pub(super) fn build(plan: &RunPlan, device: &'d ComputeDevice<B>) -> Result<Self, String> {
        let tensors = plan
            .tensors()
            .iter()
            .map(|spec| spec.build(device))
            .collect::<Result<_, _>>()?;
        let tensor_ids: Vec<TensorId> = plan.tensors().iter().map(|t| t.id()).collect();
        let nodes = plan
            .nodes()
            .iter()
            .map(|spec| spec.build(device))
            .collect::<Result<_, _>>()?;
        let node_ids: Vec<NodeId> = plan.nodes().iter().map(|n| n.id()).collect();
        let slots = plan
            .nodes()
            .iter()
            .map(|spec| {
                spec.bindings()
                    .iter()
                    .map(|b| index_of(&tensor_ids, b.tensor, "tensor"))
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        let sources = plan.nodes().iter().map(|n| n.meta().to_vec()).collect();
        let order = plan
            .order()
            .iter()
            .map(|&id| index_of(&node_ids, id, "node"))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            device,
            tensors,
            tensor_ids,
            nodes,
            node_ids,
            slots,
            sources,
            order,
        })
    }

    /// The tensor built from the spec with `id`.
    pub fn tensor(&self, id: TensorId) -> Result<&Tensor<'d, B>, String> {
        Ok(&self.tensors[index_of(&self.tensor_ids, id, "tensor")?])
    }

    /// The node built from the spec with `id`: its shader, meta and
    /// workgroups.
    pub fn node(&self, id: NodeId) -> Result<&Node<'d, B>, String> {
        Ok(&self.nodes[index_of(&self.node_ids, id, "node")?])
    }

    /// The order the nodes run in.
    pub fn order(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.order.iter().map(|&i| self.node_ids[i])
    }

    /// Lends every tensor and node to `f` as `&mut`, through an [`Edit`], to
    /// change them or put others in their place between runs. Fails if `f`
    /// does, or if it left a slot with a tensor of another kind than the
    /// slot's.
    pub fn edit<R>(
        &mut self,
        f: impl FnOnce(&mut Edit<'_, 'd, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        let out = f(&mut Edit { runnable: self });
        for (node, slots) in self.nodes.iter().zip(&self.slots) {
            let shader = node.shader();
            if shader.layout.len() != slots.len() {
                return Err(format!(
                    "edit: shader `{}` has {} slots, the plan binds {}",
                    shader.name,
                    shader.layout.len(),
                    slots.len()
                ));
            }
            for (role, &t) in shader.layout.iter().zip(slots) {
                if self.tensors[t].kind() != role_kind(role) {
                    return Err(format!(
                        "edit: shader `{}` takes {:?} where its tensor is now {:?}",
                        shader.name,
                        role_kind(role),
                        self.tensors[t].kind()
                    ));
                }
            }
        }
        out
    }

    /// Binds every tensor and node into `s` until it ends. The handle runs
    /// them, as many times as needed. Fails if a tensor or node belongs to
    /// another device than the scope's.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<BoundRunnable<'scope, 'env, 's, B>, String> {
        let metas = self.nodes.iter().map(|n| n.meta().to_vec()).collect();
        let layouts = self.nodes.iter().map(|n| n.shader().layout).collect();

        Ok(BoundRunnable {
            scope: s,
            tensors: self
                .tensors
                .iter_mut()
                .map(|t| s.bind(t))
                .collect::<Result<_, _>>()?,
            nodes: self
                .nodes
                .iter_mut()
                .map(|n| s.bind_node(n))
                .collect::<Result<_, _>>()?,
            metas,
            layouts,
            tensor_ids: &self.tensor_ids,
            node_ids: &self.node_ids,
            slots: &self.slots,
            sources: &self.sources,
            order: &self.order,
        })
    }
}

/// What [`Runnable::edit`] lends: each tensor and node as `&mut`.
pub struct Edit<'a, 'd, B: Backend + Topology> {
    runnable: &'a mut Runnable<'d, B>,
}

impl<'d, B: Backend + Topology> Edit<'_, 'd, B> {
    /// The tensor built from the spec with `id`. A tensor put in its place
    /// must be of its kind; one of another device fails the next bind, one
    /// too short the next run.
    pub fn tensor(&mut self, id: TensorId) -> Result<&mut Tensor<'d, B>, String> {
        let i = index_of(&self.runnable.tensor_ids, id, "tensor")?;
        Ok(&mut self.runnable.tensors[i])
    }

    /// The node built from the spec with `id`. A node put in its place must
    /// take tensors of the same kinds. A run still puts its `Dynamic`s'
    /// values in the per-run fields of its meta.
    pub fn node(&mut self, id: NodeId) -> Result<&mut Node<'d, B>, String> {
        let i = index_of(&self.runnable.node_ids, id, "node")?;
        Ok(&mut self.runnable.nodes[i])
    }
}

/// A [`Runnable`] bound into a scope by [`Runnable::bind`].
pub struct BoundRunnable<'scope, 'env, 's, B: Backend + Topology> {
    scope: &'scope Scope<'scope, 'env, 's, B>,
    tensors: Vec<TensorSlot<'scope>>,
    nodes: Vec<NodeSlot<'scope>>,
    // Each node's meta when bound. Only this handle changes it while the node
    // is bound, and only its per-run fields, so the rest stays as here.
    metas: Vec<Vec<MetaValue>>,
    // Each node's shader layout, read when bound: a bound node can't change.
    layouts: Vec<&'static [BindingRole]>,
    tensor_ids: &'scope [TensorId],
    node_ids: &'scope [NodeId],
    slots: &'scope [Vec<usize>],
    sources: &'scope [Vec<MetaSource>],
    order: &'scope [usize],
}

impl<'scope, 'env, 's, B: Backend + Topology> BoundRunnable<'scope, 'env, 's, B> {
    /// The slot of the tensor built from the spec with `id`, for the scope's
    /// own `run`s, `upload`s and `download`s.
    pub fn tensor(&self, id: TensorId) -> Result<TensorSlot<'scope>, String> {
        Ok(self.tensors[index_of(self.tensor_ids, id, "tensor")?])
    }

    /// Queues one run: every node, in order. Doesn't wait for it.
    ///
    /// First reads each [`Dynamic`](super::deferred::Dynamic) once, however
    /// many nodes use it, and sets the meta of the nodes that use one. Fails
    /// before anything is queued, reading none of them, if one wasn't set
    /// since the last run. A node that fails its checks stops the run; the
    /// nodes before it are queued.
    pub fn run(&self) -> Result<(), String> {
        self.apply_dynamics()?;
        self.run_order()
    }

    /// One run whose nodes `f` queues itself, in any order and any number of
    /// times, with the scope's own work between them. The `Dynamic`s are
    /// read once, before `f`, like [`BoundRunnable::run`].
    pub fn stream<R>(
        &self,
        f: impl FnOnce(&Stream<'_, 'scope, 'env, 's, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        self.apply_dynamics()?;
        f(&Stream { bound: self })
    }

    fn run_node(&self, i: usize) -> Result<(), String> {
        let args: Vec<Arg> = self.slots[i]
            .iter()
            .zip(self.layouts[i])
            .map(|(&t, role)| match role {
                BindingRole::Input(_) => Arg::Read(self.tensors[t]),
                _ => Arg::Write(self.tensors[t]),
            })
            .collect();
        self.scope.run(self.nodes[i], &args)
    }

    fn run_order(&self) -> Result<(), String> {
        self.order.iter().try_for_each(|&i| self.run_node(i))
    }

    fn apply_dynamics(&self) -> Result<(), String> {
        for (i, meta) in read_dynamics(self.sources, &self.metas)? {
            self.scope.update_meta(self.nodes[i], &meta)?;
        }
        Ok(())
    }
}

// This run's meta for every node with a per-run field: `metas` (one per node)
// with the per-run fields replaced. Every `Dynamic` is checked first, so a
// failure reads none of them.
fn read_dynamics(
    sources: &[Vec<MetaSource>],
    metas: &[Vec<MetaValue>],
) -> Result<Vec<(usize, Vec<MetaValue>)>, String> {
    let per_run = || sources.iter().flatten().filter(|s| s.is_per_run());
    let mut values = HashMap::new();
    for source in per_run() {
        let (key, value) = match source {
            MetaSource::UintPerRun(d) => (d.key(), MetaValue::Uint(d.peek_for_run()?)),
            MetaSource::FloatPerRun(d) => (d.key(), MetaValue::Float(d.peek_for_run()?)),
            _ => unreachable!("filtered to per-run sources"),
        };
        values.insert(key, value);
    }
    let mut read = HashSet::new();
    for source in per_run() {
        match source {
            MetaSource::UintPerRun(d) if read.insert(d.key()) => {
                d.read_for_run()?;
            }
            MetaSource::FloatPerRun(d) if read.insert(d.key()) => {
                d.read_for_run()?;
            }
            _ => {}
        }
    }
    Ok(sources
        .iter()
        .zip(metas)
        .enumerate()
        .filter(|(_, (s, _))| s.iter().any(MetaSource::is_per_run))
        .map(|(i, (s, meta))| {
            let meta = s
                .iter()
                .zip(meta)
                .map(|(source, now)| match source {
                    MetaSource::UintPerRun(d) => values[&d.key()],
                    MetaSource::FloatPerRun(d) => values[&d.key()],
                    _ => *now,
                })
                .collect();
            (i, meta)
        })
        .collect())
}

/// One run of a [`BoundRunnable`], queued node by node.
// TODO: changing meta and workgroups between nodes, once its design is settled.
pub struct Stream<'a, 'scope, 'env, 's, B: Backend + Topology> {
    bound: &'a BoundRunnable<'scope, 'env, 's, B>,
}

impl<B: Backend + Topology> Stream<'_, '_, '_, '_, B> {
    /// Queues the node built from the spec with `id`.
    pub fn run_node(&self, id: NodeId) -> Result<(), String> {
        self.bound
            .run_node(index_of(self.bound.node_ids, id, "node")?)
    }
}

impl<'d, B: Backend + Topology + Capturable> Runnable<'d, B> {
    /// Records one run, every node in order, to replay it. Nothing runs
    /// while recording. On a fail, gives the runnable back with the error.
    // The runnable comes back by value on a fail; a capture is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn capture(mut self) -> Result<CapturedRunnable<'d, B>, (Self, String)> {
        let device = self.device;
        let recorded = device.scope(|s| {
            let bound = self.bind(s)?;
            s.capture(|| bound.run_order())
        });
        match recorded {
            Ok(recording) => Ok(CapturedRunnable {
                runnable: self,
                recording,
            }),
            Err(e) => Err((self, e)),
        }
    }
}

/// A [`Runnable`] whose run is recorded once and replayed. Its tensors stay
/// where they were recorded: only their contents and the `Dynamic` meta
/// change between replays.
pub struct CapturedRunnable<'d, B: Backend + Topology + Capturable> {
    runnable: Runnable<'d, B>,
    recording: Recording<'d, B>,
}

impl<'d, B: Backend + Topology + Capturable> CapturedRunnable<'d, B> {
    /// [`Runnable::tensor`].
    pub fn tensor(&self, id: TensorId) -> Result<&Tensor<'d, B>, String> {
        self.runnable.tensor(id)
    }

    /// [`Runnable::node`].
    pub fn node(&self, id: NodeId) -> Result<&Node<'d, B>, String> {
        self.runnable.node(id)
    }

    /// [`Runnable::order`].
    pub fn order(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.runnable.order()
    }

    /// [`Tensor::upload`] on the tensor built from the spec with `id`.
    pub fn upload(&mut self, id: TensorId, data: &HostData) -> Result<(), String> {
        let i = index_of(&self.runnable.tensor_ids, id, "tensor")?;
        self.runnable.tensors[i].upload(data)
    }

    /// [`Tensor::zero`] on the tensor built from the spec with `id`.
    pub fn zero(&mut self, id: TensorId) -> Result<(), String> {
        let i = index_of(&self.runnable.tensor_ids, id, "tensor")?;
        self.runnable.tensors[i].zero()
    }

    /// Drops the recording and gives back the [`Runnable`].
    pub fn release(self) -> Runnable<'d, B> {
        self.runnable
    }

    /// [`Runnable::bind`], for replays.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<BoundCaptured<'scope, 'env, 's, 'd, B>, String> {
        Ok(BoundCaptured {
            bound: self.runnable.bind(s)?,
            recording: &self.recording,
        })
    }
}

/// A [`CapturedRunnable`] bound into a scope by [`CapturedRunnable::bind`].
pub struct BoundCaptured<'scope, 'env, 's, 'd, B: Backend + Topology + Capturable> {
    bound: BoundRunnable<'scope, 'env, 's, B>,
    recording: &'scope Recording<'d, B>,
}

impl<'scope, B: Backend + Topology + Capturable> BoundCaptured<'scope, '_, '_, '_, B> {
    /// [`BoundRunnable::tensor`].
    pub fn tensor(&self, id: TensorId) -> Result<TensorSlot<'scope>, String> {
        self.bound.tensor(id)
    }

    /// Queues one replay of the recorded run. Reads the `Dynamic`s like
    /// [`BoundRunnable::run`]; each node replays with its meta then.
    pub fn replay(&self) -> Result<(), String> {
        self.bound.apply_dynamics()?;
        self.bound.scope.replay(self.recording)
    }
}
