//! [`Runnable`]: a built [`RunPlan`], run inside the caller's scope. On a
//! [`Capturable`] backend it can become a [`CapturedRunnable`], which replays
//! one recording of the whole run.
//!
//! Neither opens a scope to run: [`Runnable::bind`] and
//! [`CapturedRunnable::bind`] bind everything into the caller's scope once,
//! and the handle they return runs as many times as needed in it.

use std::collections::HashMap;

use crate::backend::dtype::HostData;
use crate::backend::shader::BindingRole;
use crate::backend::{Backend, Capturable, Topology};
use crate::error::Error;
use crate::safe::{
    rules, Arg, ComputeDevice, MetaValue, Node, NodeSlot, Recording, Scope, Tensor, TensorSlot,
};
use crate::tools::plan::call::{Call, CallSpec};
use crate::tools::plan::id::GlobalId;
use crate::tools::plan::node::MetaSource;
use crate::tools::plan::run::RunPlan;
use crate::tools::plan::spec::Spec;
use crate::tools::plan::tensor::TensorSpec;

/// A built [`RunPlan`]: its tensors and calls on one device, and the order
/// the calls run in. Everything in it can be read; [`Runnable::edit`] changes
/// it between runs, [`Runnable::bind`] runs it inside a scope.
pub struct Runnable<'d, B: Backend + Topology> {
    device: &'d ComputeDevice<B>,
    tensors: HashMap<GlobalId<TensorSpec>, Tensor<'d, B>>,
    calls: HashMap<GlobalId<CallSpec>, Call<'d, B>>,
    // The calls' ids, in the order they run.
    order: Vec<GlobalId<CallSpec>>,
}

impl<'d, B: Backend + Topology> Runnable<'d, B> {
    // `RunPlan::build`, once the plan is ready.
    pub(super) fn build(plan: &RunPlan, device: &'d ComputeDevice<B>) -> Result<Self, Error> {
        let runnable = Self {
            device,
            tensors: plan
                .tensors()
                .iter()
                .map(|spec| Ok((spec.id(), spec.build(device)?)))
                .collect::<Result<_, Error>>()?,
            calls: plan
                .calls()
                .iter()
                .map(|spec| Ok((spec.id(), spec.build(device)?)))
                .collect::<Result<_, Error>>()?,
            order: plan.calls().iter().map(|c| c.id()).collect(),
        };
        for call in runnable.calls.values() {
            let shader = call.node.shader();
            for (slot, (&id, &needs)) in call.tensors.iter().zip(call.node.min_lens()).enumerate() {
                rules::long_enough(shader, slot, needs, runnable.tensor(id)?.len())?;
            }
        }
        Ok(runnable)
    }

    /// The device it was built on.
    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.device
    }

    /// Every tensor, with the id of the spec it was built from, in no order.
    pub fn tensors(&self) -> impl Iterator<Item = (GlobalId<TensorSpec>, &Tensor<'d, B>)> {
        self.tensors.iter().map(|(&id, tensor)| (id, tensor))
    }

    /// The tensor built from the spec with `id`.
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<&Tensor<'d, B>, Error> {
        self.tensors
            .get(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })
    }

    /// Every call, with the id of the spec it was built from, in no order;
    /// [`Runnable::order`] gives the order they run in.
    pub fn calls(&self) -> impl Iterator<Item = (GlobalId<CallSpec>, &Call<'d, B>)> {
        self.calls.iter().map(|(&id, call)| (id, call))
    }

    /// The call built from the spec with `id`: its node, tensors and meta
    /// sources.
    pub fn call(&self, id: GlobalId<CallSpec>) -> Result<&Call<'d, B>, Error> {
        self.calls.get(&id).ok_or(Error::NotInPlan { what: "call" })
    }

    /// The calls' ids, in the order they run.
    pub fn order(&self) -> &[GlobalId<CallSpec>] {
        &self.order
    }

    /// Lends every tensor and every call's node to `f` as `&mut`, through an
    /// [`Edit`], to change them or put others in their place between runs.
    /// [`Tensor::replace`] and [`Node::replace`] check what they put in place
    /// before anything changes. Anything put in place another way is checked
    /// only when `f` returns, and stays even if it fails: `edit` fails if `f`
    /// does, if it left a tensor or node of another device, or a call whose
    /// node doesn't fit it: slots against its tensors in count or kind, meta
    /// fields against its meta sources in count or type.
    pub fn edit<R>(
        &mut self,
        f: impl FnOnce(&mut Edit<'_, 'd, B>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let out = f(&mut Edit { runnable: self });
        for tensor in self.tensors.values() {
            rules::same_device(self.device, tensor.device(), "tensor")?;
        }
        for call in self.calls.values() {
            rules::same_device(self.device, call.node.device(), "node")?;
            let shader = call.node.shader();
            let kinds = call
                .tensors
                .iter()
                .map(|&id| Ok(self.tensor(id)?.kind()))
                .collect::<Result<Vec<_>, Error>>()?;
            rules::slot_kinds(shader, kinds.into_iter())?;
            rules::meta_types(shader, call.sources.iter().map(MetaSource::ty))?;
        }
        out
    }

    /// Binds every tensor and node into `s` until it ends. The handle runs
    /// them, as many times as needed. Fails if a tensor or node belongs to
    /// another device than the scope's.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<Bound<'scope, 'env, 's, B>, Error> {
        let tensors: HashMap<GlobalId<TensorSpec>, TensorSlot<'scope>> = self
            .tensors
            .iter_mut()
            .map(|(&id, tensor)| Ok((id, s.bind(tensor)?)))
            .collect::<Result<_, Error>>()?;
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
                    let slot = *tensors.get(id).ok_or(Error::NotInPlan { what: "tensor" })?;
                    Ok(match role {
                        BindingRole::Input(_) => Arg::Read(slot),
                        _ => Arg::Write(slot),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let meta = node.meta().to_vec();
            let slot = s.bind_node(node)?;
            let sources: &'scope [MetaSource] = sources;
            if sources.iter().any(MetaSource::is_per_run) {
                per_run.push((slot, meta, sources));
            }
            calls.insert(id, (slot, args));
        }
        Ok(Bound {
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
    /// must be of its kind and device; one too short fails the next run.
    pub fn tensor(&mut self, id: GlobalId<TensorSpec>) -> Result<&mut Tensor<'d, B>, Error> {
        self.runnable
            .tensors
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })
    }

    /// The node of the call built from the spec with `id`. A node put in its
    /// place must be of its device, take tensors of the same kinds and meta
    /// fields of the same types. [`Bound::apply_dynamics`] still sets the
    /// per-run fields of its meta, by position: a field that means something
    /// else there gets the same `Dynamic`.
    pub fn node(&mut self, id: GlobalId<CallSpec>) -> Result<&mut Node<'d, B>, Error> {
        let call = self
            .runnable
            .calls
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "call" })?;
        Ok(&mut call.node)
    }
}

/// A [`Runnable`] bound into a scope by [`Runnable::bind`].
pub struct Bound<'scope, 'env, 's, B: Backend + Topology> {
    scope: &'scope Scope<'scope, 'env, 's, B>,
    tensors: HashMap<GlobalId<TensorSpec>, TensorSlot<'scope>>,
    // Each call's node and its args.
    calls: HashMap<GlobalId<CallSpec>, (NodeSlot<'scope>, Vec<Arg<'scope>>)>,
    // The calls with a per-run meta field: the node, its meta when bound, and
    // where each field comes from. Only this handle changes a bound node's
    // meta, and only its per-run fields, so the rest stays as here.
    per_run: Vec<(NodeSlot<'scope>, Vec<MetaValue>, &'scope [MetaSource])>,
    order: &'scope [GlobalId<CallSpec>],
}

impl<'scope, 'env, 's, B: Backend + Topology> Bound<'scope, 'env, 's, B> {
    /// The slot of the tensor built from the spec with `id`, for the scope's
    /// own `run`s, `upload`s and `download`s.
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<TensorSlot<'scope>, Error> {
        Ok(*self
            .tensors
            .get(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?)
    }

    /// Queues every call, in order. Doesn't wait for them. Each node runs
    /// with its meta as it is: [`Bound::apply_dynamics`] is what changes it.
    /// A call that fails its checks stops the run with [`Error::InCall`],
    /// naming its place in [`Runnable::order`]; the calls before it are
    /// queued.
    pub fn run(&self) -> Result<(), Error> {
        for (index, &id) in self.order.iter().enumerate() {
            self.run_call(id).map_err(|error| Error::InCall {
                index,
                error: Box::new(error),
            })?;
        }
        Ok(())
    }

    /// Queues the call built from the spec with `id`, like one call of
    /// [`Bound::run`].
    pub fn run_call(&self, id: GlobalId<CallSpec>) -> Result<(), Error> {
        let (node, args) = self
            .calls
            .get(&id)
            .ok_or(Error::NotInPlan { what: "call" })?;
        self.scope.run(*node, args)
    }

    /// Sets the meta of the nodes that use a
    /// [`Dynamic`](super::deferred::Dynamic) to its value now; the runs queued
    /// after it use it. Fails, changing no meta, if one was never set.
    pub fn apply_dynamics(&self) -> Result<(), Error> {
        for (node, meta) in read_dynamics(&self.per_run)? {
            self.scope.update_meta(node, &meta)?;
        }
        Ok(())
    }
}

// The meta for each node with a per-run field: its meta when bound, with the
// per-run fields set to their `Dynamic`'s value now. Every value is read
// before any meta is set, so a failure sets none.
fn read_dynamics<'scope>(
    per_run: &[(NodeSlot<'scope>, Vec<MetaValue>, &[MetaSource])],
) -> Result<Vec<(NodeSlot<'scope>, Vec<MetaValue>)>, Error> {
    per_run
        .iter()
        .map(|(node, meta, sources)| {
            let meta = sources
                .iter()
                .zip(meta)
                .map(|(source, now)| {
                    Ok(match source {
                        MetaSource::UintPerRun(d) => MetaValue::Uint(d.read()?),
                        MetaSource::FloatPerRun(d) => MetaValue::Float(d.read()?),
                        _ => *now,
                    })
                })
                .collect::<Result<_, Error>>()?;
            Ok((*node, meta))
        })
        .collect()
}

impl<'d, B: Backend + Topology + Capturable> Runnable<'d, B> {
    /// Records one run, every call in order, to replay it. Nothing runs
    /// while recording. On a fail, gives the runnable back with the error.
    // The runnable comes back by value on a fail; a capture is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn capture(mut self) -> Result<CapturedRunnable<'d, B>, (Self, Error)> {
        let device = self.device;
        let recorded = device.scope(|s| {
            let bound = self.bind(s)?;
            s.capture(|| bound.run())
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
    // Declared first, so the recording is released while its nodes and tensors still live.
    recording: Recording<'d, B>,
    runnable: Runnable<'d, B>,
}

impl<'d, B: Backend + Topology + Capturable> CapturedRunnable<'d, B> {
    /// [`Runnable::device`].
    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.runnable.device()
    }

    /// [`Runnable::tensors`].
    pub fn tensors(&self) -> impl Iterator<Item = (GlobalId<TensorSpec>, &Tensor<'d, B>)> {
        self.runnable.tensors()
    }

    /// [`Runnable::tensor`].
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<&Tensor<'d, B>, Error> {
        self.runnable.tensor(id)
    }

    /// [`Runnable::calls`].
    pub fn calls(&self) -> impl Iterator<Item = (GlobalId<CallSpec>, &Call<'d, B>)> {
        self.runnable.calls()
    }

    /// [`Runnable::call`].
    pub fn call(&self, id: GlobalId<CallSpec>) -> Result<&Call<'d, B>, Error> {
        self.runnable.call(id)
    }

    /// [`Runnable::order`].
    pub fn order(&self) -> &[GlobalId<CallSpec>] {
        self.runnable.order()
    }

    /// [`Tensor::upload`] on the tensor built from the spec with `id`.
    pub fn upload(&mut self, id: GlobalId<TensorSpec>, data: &HostData) -> Result<(), Error> {
        self.tensor_mut(id)?.upload(data)
    }

    /// [`Tensor::zero`] on the tensor built from the spec with `id`.
    pub fn zero(&mut self, id: GlobalId<TensorSpec>) -> Result<(), Error> {
        self.tensor_mut(id)?.zero()
    }

    // Private: a tensor put in another's place would move it from where it
    // was recorded.
    fn tensor_mut(&mut self, id: GlobalId<TensorSpec>) -> Result<&mut Tensor<'d, B>, Error> {
        self.runnable
            .tensors
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })
    }

    /// Drops the recording and gives back the [`Runnable`].
    pub fn release(self) -> Runnable<'d, B> {
        self.runnable
    }

    /// [`Runnable::bind`], for replays.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<CapturedBound<'scope, 'env, 's, 'd, B>, Error> {
        Ok(CapturedBound {
            bound: self.runnable.bind(s)?,
            recording: &self.recording,
        })
    }
}

/// A [`CapturedRunnable`] bound into a scope by [`CapturedRunnable::bind`].
pub struct CapturedBound<'scope, 'env, 's, 'd, B: Backend + Topology + Capturable> {
    bound: Bound<'scope, 'env, 's, B>,
    recording: &'scope Recording<'d, B>,
}

impl<'scope, B: Backend + Topology + Capturable> CapturedBound<'scope, '_, '_, '_, B> {
    /// [`Bound::tensor`].
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<TensorSlot<'scope>, Error> {
        self.bound.tensor(id)
    }

    /// [`Bound::apply_dynamics`].
    pub fn apply_dynamics(&self) -> Result<(), Error> {
        self.bound.apply_dynamics()
    }

    /// Queues one replay of the recorded run. Each node replays with its
    /// meta as it is: [`CapturedBound::apply_dynamics`] is what changes it.
    pub fn replay(&self) -> Result<(), Error> {
        self.bound.scope.replay(self.recording)
    }
}
