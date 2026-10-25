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
use crate::tools::plan::call::{Call, CallId};
use crate::tools::plan::node::MetaSource;
use crate::tools::plan::run::{role_kind, RunPlan};
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::TensorId;

/// A built [`RunPlan`]: its tensors and calls on one device, and the order
/// the calls run in. Everything in it can be read; [`Runnable::edit`] changes
/// it between runs, [`Runnable::bind`] runs it inside a scope.
pub struct Runnable<'d, B: Backend + Topology> {
    device: &'d ComputeDevice<B>,
    tensors: HashMap<TensorId, Tensor<'d, B>>,
    calls: HashMap<CallId, Call<'d, B>>,
    // The calls' ids, in the order they run.
    order: Vec<CallId>,
}

impl<'d, B: Backend + Topology> Runnable<'d, B> {
    // `RunPlan::build`, once the plan is ready.
    pub(super) fn build(plan: &RunPlan, device: &'d ComputeDevice<B>) -> Result<Self, String> {
        Ok(Self {
            device,
            tensors: plan
                .tensors()
                .iter()
                .map(|spec| Ok((spec.id(), spec.build(device)?)))
                .collect::<Result<_, String>>()?,
            calls: plan
                .calls()
                .iter()
                .map(|spec| Ok((spec.id(), spec.build(device)?)))
                .collect::<Result<_, String>>()?,
            order: plan.calls().iter().map(|c| c.id()).collect(),
        })
    }

    /// The tensor built from the spec with `id`.
    pub fn tensor(&self, id: TensorId) -> Result<&Tensor<'d, B>, String> {
        Ok(self
            .tensors
            .get(&id)
            .ok_or("this tensor isn't in the plan")?)
    }

    /// The call built from the spec with `id`: its node, tensors and meta
    /// sources.
    pub fn call(&self, id: CallId) -> Result<&Call<'d, B>, String> {
        Ok(self.calls.get(&id).ok_or("this call isn't in the plan")?)
    }

    /// The calls' ids, in the order they run.
    pub fn order(&self) -> &[CallId] {
        &self.order
    }

    /// Lends every tensor and every call's node to `f` as `&mut`, through an
    /// [`Edit`], to change them or put others in their place between runs.
    /// Fails if `f` does, or if it left a call whose node's slots don't fit
    /// its tensors in count or kind.
    pub fn edit<R>(
        &mut self,
        f: impl FnOnce(&mut Edit<'_, 'd, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        let out = f(&mut Edit { runnable: self });
        for call in self.calls.values() {
            let shader = call.node.shader();
            if shader.layout.len() != call.tensors.len() {
                return Err(format!(
                    "edit: shader `{}` has {} slots, its call gives {} tensors",
                    shader.name,
                    shader.layout.len(),
                    call.tensors.len()
                ));
            }
            for (role, &id) in shader.layout.iter().zip(&call.tensors) {
                let tensor = self.tensor(id)?;
                if tensor.kind() != role_kind(role) {
                    return Err(format!(
                        "edit: shader `{}` takes {:?} where its tensor is now {:?}",
                        shader.name,
                        role_kind(role),
                        tensor.kind()
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
        let tensors: HashMap<TensorId, TensorSlot<'scope>> = self
            .tensors
            .iter_mut()
            .map(|(&id, tensor)| Ok((id, s.bind(tensor)?)))
            .collect::<Result<_, String>>()?;
        let mut calls = HashMap::with_capacity(self.calls.len());
        let mut per_run = Vec::new();
        for (&id, call) in self.calls.iter_mut() {
            let Call {
                node,
                tensors: ids,
                sources,
            } = call;
            let args = ids
                .iter()
                .zip(node.shader().layout)
                .map(|(id, role)| {
                    let slot = *tensors.get(id).ok_or("this tensor isn't in the plan")?;
                    Ok(match role {
                        BindingRole::Input(_) => Arg::Read(slot),
                        _ => Arg::Write(slot),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let meta = node.meta().to_vec();
            let slot = s.bind_node(node)?;
            let sources: &'scope [MetaSource] = sources;
            if sources.iter().any(MetaSource::is_per_run) {
                per_run.push((slot, meta, sources));
            }
            calls.insert(id, (slot, args));
        }
        Ok(BoundRunnable {
            scope: s,
            tensors,
            calls,
            per_run,
            order: &self.order,
        })
    }
}

/// What [`Runnable::edit`] lends: each tensor and each call's node as `&mut`.
pub struct Edit<'a, 'd, B: Backend + Topology> {
    runnable: &'a mut Runnable<'d, B>,
}

impl<'d, B: Backend + Topology> Edit<'_, 'd, B> {
    /// The tensor built from the spec with `id`. A tensor put in its place
    /// must be of its kind; one of another device fails the next bind, one
    /// too short the next run.
    pub fn tensor(&mut self, id: TensorId) -> Result<&mut Tensor<'d, B>, String> {
        Ok(self
            .runnable
            .tensors
            .get_mut(&id)
            .ok_or("this tensor isn't in the plan")?)
    }

    /// The node of the call built from the spec with `id`. A node put in its
    /// place must take tensors of the same kinds. A run still puts its
    /// `Dynamic`s' values in the per-run fields of its meta.
    pub fn node(&mut self, id: CallId) -> Result<&mut Node<'d, B>, String> {
        let call = self
            .runnable
            .calls
            .get_mut(&id)
            .ok_or("this call isn't in the plan")?;
        Ok(&mut call.node)
    }
}

/// A [`Runnable`] bound into a scope by [`Runnable::bind`].
pub struct BoundRunnable<'scope, 'env, 's, B: Backend + Topology> {
    scope: &'scope Scope<'scope, 'env, 's, B>,
    tensors: HashMap<TensorId, TensorSlot<'scope>>,
    // Each call's node and its args.
    calls: HashMap<CallId, (NodeSlot<'scope>, Vec<Arg<'scope>>)>,
    // The calls with a per-run meta field: the node, its meta when bound, and
    // where each field comes from. Only this handle changes a bound node's
    // meta, and only its per-run fields, so the rest stays as here.
    per_run: Vec<(NodeSlot<'scope>, Vec<MetaValue>, &'scope [MetaSource])>,
    order: &'scope [CallId],
}

impl<'scope, 'env, 's, B: Backend + Topology> BoundRunnable<'scope, 'env, 's, B> {
    /// The slot of the tensor built from the spec with `id`, for the scope's
    /// own `run`s, `upload`s and `download`s.
    pub fn tensor(&self, id: TensorId) -> Result<TensorSlot<'scope>, String> {
        Ok(*self
            .tensors
            .get(&id)
            .ok_or("this tensor isn't in the plan")?)
    }

    /// Queues one run: every call, in order. Doesn't wait for it.
    ///
    /// First reads each [`Dynamic`](super::deferred::Dynamic) once, however
    /// many nodes use it, and sets the meta of the nodes that use one. Fails
    /// before anything is queued, reading none of them, if one wasn't set
    /// since the last run. A call that fails its checks stops the run; the
    /// calls before it are queued.
    pub fn run(&self) -> Result<(), String> {
        self.apply_dynamics()?;
        self.run_all()
    }

    /// One run whose calls `f` queues itself, in any order and any number of
    /// times, with the scope's own work between them. The `Dynamic`s are
    /// read once, before `f`, like [`BoundRunnable::run`].
    pub fn stream<R>(
        &self,
        f: impl FnOnce(&Stream<'_, 'scope, 'env, 's, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        self.apply_dynamics()?;
        f(&Stream { bound: self })
    }

    fn run_call(&self, id: CallId) -> Result<(), String> {
        let (node, args) = self.calls.get(&id).ok_or("this call isn't in the plan")?;
        self.scope.run(*node, args)
    }

    fn run_all(&self) -> Result<(), String> {
        self.order.iter().try_for_each(|&id| self.run_call(id))
    }

    fn apply_dynamics(&self) -> Result<(), String> {
        for (node, meta) in read_dynamics(&self.per_run)? {
            self.scope.update_meta(node, &meta)?;
        }
        Ok(())
    }
}

// This run's meta for each node with a per-run field: its meta when bound,
// with the per-run fields replaced. Every `Dynamic` is checked first, so a
// failure reads none of them.
fn read_dynamics<'scope>(
    per_run: &[(NodeSlot<'scope>, Vec<MetaValue>, &[MetaSource])],
) -> Result<Vec<(NodeSlot<'scope>, Vec<MetaValue>)>, String> {
    let dynamics = || {
        per_run
            .iter()
            .flat_map(|(_, _, sources)| sources.iter())
            .filter(|s| s.is_per_run())
    };
    let mut values = HashMap::new();
    for source in dynamics() {
        let (key, value) = match source {
            MetaSource::UintPerRun(d) => (d.key(), MetaValue::Uint(d.peek_for_run()?)),
            MetaSource::FloatPerRun(d) => (d.key(), MetaValue::Float(d.peek_for_run()?)),
            _ => unreachable!("filtered to per-run sources"),
        };
        values.insert(key, value);
    }
    let mut read = HashSet::new();
    for source in dynamics() {
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
    Ok(per_run
        .iter()
        .map(|(node, meta, sources)| {
            let meta = sources
                .iter()
                .zip(meta)
                .map(|(source, now)| match source {
                    MetaSource::UintPerRun(d) => values[&d.key()],
                    MetaSource::FloatPerRun(d) => values[&d.key()],
                    _ => *now,
                })
                .collect();
            (*node, meta)
        })
        .collect())
}

/// One run of a [`BoundRunnable`], queued call by call.
// TODO: changing meta and workgroups between calls, once its design is settled.
pub struct Stream<'a, 'scope, 'env, 's, B: Backend + Topology> {
    bound: &'a BoundRunnable<'scope, 'env, 's, B>,
}

impl<B: Backend + Topology> Stream<'_, '_, '_, '_, B> {
    /// Queues the call built from the spec with `id`.
    pub fn run_call(&self, id: CallId) -> Result<(), String> {
        self.bound.run_call(id)
    }
}

impl<'d, B: Backend + Topology + Capturable> Runnable<'d, B> {
    /// Records one run, every call in order, to replay it. Nothing runs
    /// while recording. On a fail, gives the runnable back with the error.
    // The runnable comes back by value on a fail; a capture is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn capture(mut self) -> Result<CapturedRunnable<'d, B>, (Self, String)> {
        let device = self.device;
        let recorded = device.scope(|s| {
            let bound = self.bind(s)?;
            s.capture(|| bound.run_all())
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

    /// [`Runnable::call`].
    pub fn call(&self, id: CallId) -> Result<&Call<'d, B>, String> {
        self.runnable.call(id)
    }

    /// [`Runnable::order`].
    pub fn order(&self) -> &[CallId] {
        self.runnable.order()
    }

    /// [`Tensor::upload`] on the tensor built from the spec with `id`.
    pub fn upload(&mut self, id: TensorId, data: &HostData) -> Result<(), String> {
        self.tensor_mut(id)?.upload(data)
    }

    /// [`Tensor::zero`] on the tensor built from the spec with `id`.
    pub fn zero(&mut self, id: TensorId) -> Result<(), String> {
        self.tensor_mut(id)?.zero()
    }

    // Private: a tensor put in another's place would move it from where it
    // was recorded.
    fn tensor_mut(&mut self, id: TensorId) -> Result<&mut Tensor<'d, B>, String> {
        Ok(self
            .runnable
            .tensors
            .get_mut(&id)
            .ok_or("this tensor isn't in the plan")?)
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
