//! [`Runnable`]: a built [`RunPlan`], run inside the caller's scope. On a
//! [`Capturable`] backend it can become a [`CapturedRunnable`], which replays
//! one recording of the whole run.
//!
//! Neither opens a scope to run: [`Runnable::bind`] and
//! [`CapturedRunnable::bind`] bind everything into the caller's scope once,
//! and the handle they return runs as many times as needed in it.

use std::collections::{HashMap, HashSet};

use crate::backend::dtype::{DataKind, HostData};
use crate::backend::shader::{BindingRole, Shader};
use crate::backend::{Backend, Capturable, Topology};
use crate::safe::{
    Arg, ComputeDevice, MetaValue, Node, NodeSlot, Recording, Scope, Tensor, TensorSlot,
};
use crate::tools::plan::node::{MetaSource, NodeId};
use crate::tools::plan::run::{RunPart, RunPlan};
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::TensorId;

// A slot's tensor, by its index in `Runnable::tensors`.
enum PlanArg {
    Read(usize),
    Write(usize),
}

struct NodeWiring {
    // One per slot, in slot order.
    args: Vec<PlanArg>,
    sources: Vec<MetaSource>,
}

// Everything but the tensors and nodes themselves, so a bound handle can read
// it while they are lent to the scope.
struct Wiring {
    tensor_index: HashMap<TensorId, usize>,
    // Each tensor's kind as built; an edit can't change it.
    kinds: Vec<DataKind>,
    node_index: HashMap<NodeId, usize>,
    // Each node's shader as built; an edit can't change it.
    shaders: Vec<&'static Shader>,
    // One per node, by index in `Runnable::nodes`.
    nodes: Vec<NodeWiring>,
    parts: Vec<RunPart>,
    // Each part's nodes, by index in `Runnable::nodes`.
    part_nodes: Vec<Vec<usize>>,
}

impl Wiring {
    fn tensor(&self, id: TensorId) -> Result<usize, String> {
        self.tensor_index
            .get(&id)
            .copied()
            .ok_or_else(|| "this tensor isn't in the plan".to_string())
    }

    fn node(&self, id: NodeId) -> Result<usize, String> {
        self.node_index
            .get(&id)
            .copied()
            .ok_or_else(|| "this node isn't in the plan".to_string())
    }

    // This run's meta for every node with a per-run field: `metas` (one per
    // node) with the per-run fields replaced. Every `Dynamic` is checked
    // first, so a failure reads none of them.
    fn read_dynamics(
        &self,
        metas: &[Vec<MetaValue>],
    ) -> Result<Vec<(usize, Vec<MetaValue>)>, String> {
        let per_run = || {
            self.nodes
                .iter()
                .flat_map(|n| &n.sources)
                .filter(|s| s.is_per_run())
        };
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
        Ok(self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.sources.iter().any(MetaSource::is_per_run))
            .map(|(i, n)| {
                let meta = n
                    .sources
                    .iter()
                    .zip(&metas[i])
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
}

/// A built [`RunPlan`]: its tensors and nodes on one device, run part after
/// part. Everything in it can be read; [`Runnable::edit`] changes it between
/// runs, [`Runnable::bind`] runs it inside a scope.
pub struct Runnable<'d, B: Backend + Topology> {
    device: &'d ComputeDevice<B>,
    tensors: Vec<Tensor<'d, B>>,
    nodes: Vec<Node<'d, B>>,
    wiring: Wiring,
}

impl<'d, B: Backend + Topology> Runnable<'d, B> {
    // `RunPlan::build`, once the plan is ready.
    pub(super) fn build(plan: &RunPlan, device: &'d ComputeDevice<B>) -> Result<Self, String> {
        let mut tensors = Vec::with_capacity(plan.tensors().len());
        let mut tensor_index = HashMap::new();
        for spec in plan.tensors() {
            tensor_index.insert(spec.id(), tensors.len());
            tensors.push(spec.build(device)?);
        }

        let mut nodes = Vec::with_capacity(plan.nodes().len());
        let mut node_index = HashMap::new();
        let mut wired = Vec::with_capacity(plan.nodes().len());
        for spec in plan.nodes() {
            node_index.insert(spec.id(), nodes.len());
            let args = spec
                .bindings()
                .iter()
                .zip(spec.shader().layout)
                .map(|(binding, role)| {
                    let index = tensor_index[&binding.tensor];
                    match role {
                        BindingRole::Input(_) => PlanArg::Read(index),
                        _ => PlanArg::Write(index),
                    }
                })
                .collect();
            nodes.push(spec.build(device)?);
            wired.push(NodeWiring {
                args,
                sources: spec.meta().to_vec(),
            });
        }

        let part_nodes = plan
            .parts()
            .iter()
            .map(|part| part.nodes.iter().map(|id| node_index[id]).collect())
            .collect();
        let kinds = tensors.iter().map(Tensor::kind).collect();
        let shaders = nodes.iter().map(Node::shader).collect();
        Ok(Runnable {
            device,
            tensors,
            nodes,
            wiring: Wiring {
                tensor_index,
                kinds,
                node_index,
                shaders,
                nodes: wired,
                parts: plan.parts().to_vec(),
                part_nodes,
            },
        })
    }

    /// The tensor built from the spec with `id`.
    pub fn tensor(&self, id: TensorId) -> Result<&Tensor<'d, B>, String> {
        Ok(&self.tensors[self.wiring.tensor(id)?])
    }

    /// The node built from the spec with `id`: its shader, meta and
    /// workgroups.
    pub fn node(&self, id: NodeId) -> Result<&Node<'d, B>, String> {
        Ok(&self.nodes[self.wiring.node(id)?])
    }

    /// The order the nodes run in.
    pub fn parts(&self) -> &[RunPart] {
        &self.wiring.parts
    }

    /// Changes tensors and nodes between runs, through an [`Edit`]. Fails if
    /// `f` does, or if it left a tensor of another kind or a node of another
    /// shader than the one built.
    pub fn edit<R>(
        &mut self,
        f: impl FnOnce(&mut Edit<'_, 'd, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        let out = f(&mut Edit { runnable: self });
        for (tensor, kind) in self.tensors.iter().zip(&self.wiring.kinds) {
            if tensor.kind() != *kind {
                return Err(format!(
                    "edit: a {kind:?} tensor was replaced by a {:?} one",
                    tensor.kind()
                ));
            }
        }
        for (node, shader) in self.nodes.iter().zip(&self.wiring.shaders) {
            if !std::ptr::eq(node.shader(), *shader) {
                return Err(format!(
                    "edit: a node of shader `{}` was replaced by one of `{}`",
                    shader.name,
                    node.shader().name
                ));
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
        let Self {
            tensors,
            nodes,
            wiring,
            ..
        } = self;
        Ok(BoundRunnable {
            bound: Bound::new(s, tensors, nodes, wiring)?,
        })
    }
}

/// What [`Runnable::edit`] lends: each tensor and node as `&mut`, to change
/// it or put another in its place.
pub struct Edit<'a, 'd, B: Backend + Topology> {
    runnable: &'a mut Runnable<'d, B>,
}

impl<'d, B: Backend + Topology> Edit<'_, 'd, B> {
    /// The tensor built from the spec with `id`. A tensor put in its place
    /// must be of its kind; one of another device fails the next bind, one
    /// too short the next run.
    pub fn tensor(&mut self, id: TensorId) -> Result<&mut Tensor<'d, B>, String> {
        let i = self.runnable.wiring.tensor(id)?;
        Ok(&mut self.runnable.tensors[i])
    }

    /// The node built from the spec with `id`. A node put in its place must
    /// run its shader. A run still puts its `Dynamic`s' values in the
    /// per-run fields of its meta.
    pub fn node(&mut self, id: NodeId) -> Result<&mut Node<'d, B>, String> {
        let i = self.runnable.wiring.node(id)?;
        Ok(&mut self.runnable.nodes[i])
    }
}

// The slots a bound runnable got from the scope, and what runs them.
struct Bound<'scope, 'env, 's, B: Backend + Topology> {
    scope: &'scope Scope<'scope, 'env, 's, B>,
    tensors: Vec<TensorSlot<'scope>>,
    nodes: Vec<NodeSlot<'scope>>,
    // Each node's meta when bound. Only this handle changes it while the node
    // is bound, and only its per-run fields, so the rest stays as here.
    metas: Vec<Vec<MetaValue>>,
    wiring: &'scope Wiring,
}

impl<'scope, 'env, 's, B: Backend + Topology> Bound<'scope, 'env, 's, B> {
    fn new<'d>(
        scope: &'scope Scope<'scope, 'env, 's, B>,
        tensors: &'scope mut [Tensor<'d, B>],
        nodes: &'scope mut [Node<'d, B>],
        wiring: &'scope Wiring,
    ) -> Result<Self, String> {
        let metas = nodes.iter().map(|n| n.meta().to_vec()).collect();
        Ok(Self {
            scope,
            metas,
            tensors: tensors
                .iter_mut()
                .map(|t| scope.bind(t))
                .collect::<Result<_, _>>()?,
            nodes: nodes
                .iter_mut()
                .map(|n| scope.bind_node(n))
                .collect::<Result<_, _>>()?,
            wiring,
        })
    }

    fn tensor(&self, id: TensorId) -> Result<TensorSlot<'scope>, String> {
        Ok(self.tensors[self.wiring.tensor(id)?])
    }

    fn apply_dynamics(&self) -> Result<(), String> {
        for (i, meta) in self.wiring.read_dynamics(&self.metas)? {
            self.scope.update_meta(self.nodes[i], &meta)?;
        }
        Ok(())
    }

    fn run_part(&self, index: usize) -> Result<(), String> {
        let part = self
            .wiring
            .part_nodes
            .get(index)
            .ok_or_else(|| format!("run_part: there is no part {index}"))?;
        for &i in part {
            let args: Vec<Arg> = self.wiring.nodes[i]
                .args
                .iter()
                .map(|arg| match *arg {
                    PlanArg::Read(t) => Arg::Read(self.tensors[t]),
                    PlanArg::Write(t) => Arg::Write(self.tensors[t]),
                })
                .collect();
            self.scope.run(self.nodes[i], &args)?;
        }
        Ok(())
    }

    fn run_parts(&self) -> Result<(), String> {
        (0..self.wiring.part_nodes.len()).try_for_each(|i| self.run_part(i))
    }
}

/// A [`Runnable`] bound into a scope by [`Runnable::bind`].
pub struct BoundRunnable<'scope, 'env, 's, B: Backend + Topology> {
    bound: Bound<'scope, 'env, 's, B>,
}

impl<'scope, 'env, 's, B: Backend + Topology> BoundRunnable<'scope, 'env, 's, B> {
    /// The slot of the tensor built from the spec with `id`, for the scope's
    /// own `run`s, `upload`s and `download`s.
    pub fn tensor(&self, id: TensorId) -> Result<TensorSlot<'scope>, String> {
        self.bound.tensor(id)
    }

    /// Queues one run: every part, in order. Doesn't wait for it.
    ///
    /// First reads each [`Dynamic`](super::deferred::Dynamic) once, however
    /// many nodes use it, and sets the meta of the nodes that use one. Fails
    /// before anything is queued, reading none of them, if one wasn't set
    /// since the last run. A node that fails its checks stops the run; the
    /// nodes before it are queued.
    pub fn run(&self) -> Result<(), String> {
        self.bound.apply_dynamics()?;
        self.bound.run_parts()
    }

    /// One run whose parts `f` queues itself, in any order and any number
    /// of times, with the scope's own work between them. The `Dynamic`s are
    /// read once, before `f`, like [`BoundRunnable::run`].
    pub fn stream<R>(
        &self,
        f: impl FnOnce(&Stream<'_, 'scope, 'env, 's, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        self.bound.apply_dynamics()?;
        f(&Stream { bound: &self.bound })
    }
}

/// One run of a [`BoundRunnable`], queued part by part.
// TODO: changing meta and workgroups between parts, once its design is settled.
pub struct Stream<'a, 'scope, 'env, 's, B: Backend + Topology> {
    bound: &'a Bound<'scope, 'env, 's, B>,
}

impl<B: Backend + Topology> Stream<'_, '_, '_, '_, B> {
    /// How many parts the plan has.
    pub fn parts(&self) -> usize {
        self.bound.wiring.part_nodes.len()
    }

    /// Queues part `index`'s nodes, in order.
    pub fn run_part(&self, index: usize) -> Result<(), String> {
        self.bound.run_part(index)
    }
}

impl<'d, B: Backend + Topology + Capturable> Runnable<'d, B> {
    /// Records one run, every part in order, to replay it. Nothing runs
    /// while recording. On a fail, gives the runnable back with the error.
    // The runnable comes back by value on a fail; a capture is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn capture(mut self) -> Result<CapturedRunnable<'d, B>, (Self, String)> {
        let device = self.device;
        let recorded = device.scope(|s| {
            let bound = self.bind(s)?;
            s.capture(|| bound.bound.run_parts())
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

    /// [`Runnable::parts`].
    pub fn parts(&self) -> &[RunPart] {
        self.runnable.parts()
    }

    /// [`Tensor::upload`] on the tensor built from the spec with `id`.
    pub fn upload(&mut self, id: TensorId, data: &HostData) -> Result<(), String> {
        let i = self.runnable.wiring.tensor(id)?;
        self.runnable.tensors[i].upload(data)
    }

    /// [`Tensor::zero`] on the tensor built from the spec with `id`.
    pub fn zero(&mut self, id: TensorId) -> Result<(), String> {
        let i = self.runnable.wiring.tensor(id)?;
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
        let Runnable {
            tensors,
            nodes,
            wiring,
            ..
        } = &mut self.runnable;
        Ok(BoundCaptured {
            bound: Bound::new(s, tensors, nodes, wiring)?,
            recording: &self.recording,
        })
    }
}

/// A [`CapturedRunnable`] bound into a scope by [`CapturedRunnable::bind`].
pub struct BoundCaptured<'scope, 'env, 's, 'd, B: Backend + Topology + Capturable> {
    bound: Bound<'scope, 'env, 's, B>,
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
