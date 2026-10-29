//! [`RunSpec`]: tensors, and the calls that run on them, in order. Built, it
//! is a [`Run`], run inside the caller's scope. On a [`Capturable`] backend
//! it can become a [`CapturedRun`], which replays one recording of the whole
//! run.
//!
//! Neither opens a scope to run: [`Run::bind`] and
//! [`CapturedRun::bind`] bind everything into the caller's scope once,
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
use crate::tools::plan::tensor::{AnyTensor, TensorSpec};

/// What a [`RunSpec`] is made from.
pub struct RunDesc {
    /// Every tensor a call of the plan names.
    pub tensors: Vec<TensorSpec>,
    /// The calls, in the order they run.
    pub calls: Vec<CallSpec>,
}

/// Tensors to build on one device, and the calls that run on them, in order.
pub struct RunSpec {
    id: GlobalId<RunSpec>,
    desc: RunDesc,
}

impl RunSpec {
    /// The tensors to build.
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.desc.tensors
    }

    /// The calls, in the order they run.
    pub fn calls(&self) -> &[CallSpec] {
        &self.desc.calls
    }

    /// Fails if a call names a tensor that isn't in the plan, one of another
    /// kind than its slot's, or a written slot's tensor in another slot too.
    pub fn new(desc: RunDesc) -> Result<Self, Error> {
        for call in &desc.calls {
            let kinds = call
                .tensors()
                .iter()
                .map(|id| {
                    let tensor = desc.tensors.iter().find(|t| t.id() == *id);
                    Ok(tensor.ok_or(Error::NotInPlan { what: "tensor" })?.kind())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            rules::slot_kinds(call.node().shader().signature(), kinds.into_iter())?;
            rules::not_aliased(call.node().shader().signature(), call.tensors())?;
        }
        Ok(Self {
            id: GlobalId::new(),
            desc,
        })
    }

    pub fn id(&self) -> GlobalId<RunSpec> {
        self.id
    }

    /// Fails where [`TensorSpec::ready`] or [`CallSpec::ready`] does.
    pub fn ready(&self) -> Result<(), Error> {
        for tensor in &self.desc.tensors {
            tensor.ready()?;
        }
        for call in &self.desc.calls {
            call.ready()?;
        }
        Ok(())
    }

    /// Builds every tensor, then every call. Also fails if a tensor is
    /// shorter than its slot needs for its call's meta. It takes the spec,
    /// so a spec makes one live run; on a fail the spec comes back with the
    /// error.
    pub fn build<'d, B: Backend + Topology>(
        self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Run<'d, B>, (Self, Error)> {
        match self.ready().and_then(|()| Run::build(&self, device)) {
            Ok(run) => Ok(run),
            Err(e) => Err((self, e)),
        }
    }

    /// A run with every call built and every tensor hollow, as many times
    /// as needed; [`Run::fill_from`] moves live tensors in. Fails where
    /// [`CallSpec::ready`] or [`TensorSpec::build_hollow`] does, or if a
    /// tensor is shorter than its slot needs for its call's meta.
    pub fn build_hollow<'d, B: Backend + Topology>(
        &self,
        device: &'d ComputeDevice<B>,
    ) -> Result<Run<'d, B>, Error> {
        for call in &self.desc.calls {
            call.ready()?;
        }
        Run::build_hollow(self, device)
    }
}

/// A built [`RunSpec`]: its tensors and calls on one device, and the order
/// the calls run in. Everything in it can be read; [`Run::edit`] changes
/// it between runs, [`Run::bind`] runs it inside a scope.
pub struct Run<'d, B: Backend + Topology> {
    device: &'d ComputeDevice<B>,
    tensors: HashMap<GlobalId<TensorSpec>, AnyTensor<'d, B>>,
    calls: HashMap<GlobalId<CallSpec>, Call<'d, B>>,
    // The calls' ids, in the order they run.
    order: Vec<GlobalId<CallSpec>>,
}

impl<'d, B: Backend + Topology> Run<'d, B> {
    // `RunSpec::build`, once the plan is ready.
    pub(super) fn build(plan: &RunSpec, device: &'d ComputeDevice<B>) -> Result<Self, Error> {
        let tensors = plan
            .tensors()
            .iter()
            .map(|spec| Ok((spec.id(), AnyTensor::Live(spec.build(device)?))))
            .collect::<Result<_, Error>>()?;
        Self::with_tensors(plan, device, tensors)
    }

    // `RunSpec::build_hollow`, once the plan's calls are ready.
    pub(super) fn build_hollow(
        plan: &RunSpec,
        device: &'d ComputeDevice<B>,
    ) -> Result<Self, Error> {
        let tensors = plan
            .tensors()
            .iter()
            .map(|spec| Ok((spec.id(), AnyTensor::Hollow(spec.build_hollow()?))))
            .collect::<Result<_, Error>>()?;
        Self::with_tensors(plan, device, tensors)
    }

    // Builds the plan's calls around `tensors`, built from its tensors.
    fn with_tensors(
        plan: &RunSpec,
        device: &'d ComputeDevice<B>,
        tensors: HashMap<GlobalId<TensorSpec>, AnyTensor<'d, B>>,
    ) -> Result<Self, Error> {
        let run = Self {
            device,
            tensors,
            calls: plan
                .calls()
                .iter()
                .map(|spec| Ok((spec.id(), spec.build(device)?)))
                .collect::<Result<_, Error>>()?,
            order: plan.calls().iter().map(|c| c.id()).collect(),
        };
        for call in run.calls.values() {
            let shader = call.node.signature();
            for (slot, (&id, &needs)) in call.tensors.iter().zip(call.node.min_lens()).enumerate() {
                rules::long_enough(shader, slot, needs, run.any_tensor(id)?.len())?;
            }
        }
        Ok(run)
    }

    /// The device it was built on.
    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.device
    }

    /// Every tensor, live or hollow, with the id of the spec it was built
    /// from, in no order.
    pub fn tensors(&self) -> impl Iterator<Item = (GlobalId<TensorSpec>, &AnyTensor<'d, B>)> {
        self.tensors.iter().map(|(&id, tensor)| (id, tensor))
    }

    /// The tensor built from the spec with `id`; [`Error::Hollow`] if it
    /// is hollow.
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<&Tensor<'d, B>, Error> {
        self.any_tensor(id)?.live()
    }

    /// The tensor built from the spec with `id`, live or hollow.
    pub fn any_tensor(&self, id: GlobalId<TensorSpec>) -> Result<&AnyTensor<'d, B>, Error> {
        self.tensors
            .get(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })
    }

    /// Whether a tensor of it is hollow; such a run doesn't bind.
    pub fn contains_hollow(&self) -> bool {
        self.tensors.values().any(AnyTensor::is_hollow)
    }

    /// Moves the live tensor built from the spec with `id` out of `from`
    /// into its hollow place here, with no copy; it is hollow in `from`
    /// after. Fails, changing nothing, if `from` is of another device, if
    /// either run lacks `id`, if it isn't hollow here or live in `from`, or
    /// if the two differ in kind or length.
    pub fn fill_from(
        &mut self,
        from: &mut Run<'d, B>,
        id: GlobalId<TensorSpec>,
    ) -> Result<(), Error> {
        rules::same_device(self.device, from.device, "tensor")?;
        let to = self
            .tensors
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?;
        let live = from
            .tensors
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?;
        if !to.is_hollow() {
            return Err(Error::NotHollow);
        }
        live.live()?;
        rules::same_kind(to.kind(), live.kind())?;
        rules::same_len(to.len(), live.len())?;
        // The live tensor moves in; the hollow one, of its kind and length, takes its place.
        std::mem::swap(to, live);
        Ok(())
    }

    /// Every call, with the id of the spec it was built from, in no order;
    /// [`Run::order`] gives the order they run in.
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
        let out = f(&mut Edit { run: self });
        for tensor in self.tensors.values() {
            if let AnyTensor::Live(tensor) = tensor {
                rules::same_device(self.device, tensor.device(), "tensor")?;
            }
        }
        for call in self.calls.values() {
            rules::same_device(self.device, call.node.device(), "node")?;
            let shader = call.node.signature();
            let kinds = call
                .tensors
                .iter()
                .map(|&id| Ok(self.any_tensor(id)?.kind()))
                .collect::<Result<Vec<_>, Error>>()?;
            rules::slot_kinds(shader, kinds.into_iter())?;
            rules::meta_types(shader, call.sources.iter().map(MetaSource::ty))?;
        }
        out
    }

    /// Binds every tensor and node into `s` until it ends. The handle runs
    /// them, as many times as needed. Fails if a tensor or node belongs to
    /// another device than the scope's, or if a tensor is hollow.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<Bound<'scope, 'env, 's, B>, Error> {
        let tensors: HashMap<GlobalId<TensorSpec>, TensorSlot<'scope>> = self
            .tensors
            .iter_mut()
            .map(|(&id, tensor)| Ok((id, s.bind(tensor.live_mut()?)?)))
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
                .zip(node.signature().layout)
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

/// What [`Run::edit`] lends: each tensor and each call's node as `&mut`.
pub struct Edit<'a, 'd, B: Backend + Topology> {
    run: &'a mut Run<'d, B>,
}

impl<'d, B: Backend + Topology> Edit<'_, 'd, B> {
    /// The tensor built from the spec with `id`; [`Error::Hollow`] if it is
    /// hollow. A tensor put in its place must be of its kind and device; one
    /// too short fails the next run.
    pub fn tensor(&mut self, id: GlobalId<TensorSpec>) -> Result<&mut Tensor<'d, B>, Error> {
        self.run
            .tensors
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?
            .live_mut()
    }

    /// The node of the call built from the spec with `id`. A node put in its
    /// place must be of its device, take tensors of the same kinds and meta
    /// fields of the same types. [`Bound::apply_dynamics`] still sets the
    /// per-run fields of its meta, by position: a field that means something
    /// else there gets the same `Dynamic`.
    pub fn node(&mut self, id: GlobalId<CallSpec>) -> Result<&mut Node<'d, B>, Error> {
        let call = self
            .run
            .calls
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "call" })?;
        Ok(&mut call.node)
    }
}

/// A [`Run`] bound into a scope by [`Run::bind`].
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
    /// own `execute`s, `upload`s and `download`s.
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<TensorSlot<'scope>, Error> {
        Ok(*self
            .tensors
            .get(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?)
    }

    /// Queues every call, in order. Doesn't wait for them. Each node runs
    /// with its meta as it is: [`Bound::apply_dynamics`] is what changes it.
    /// A call that fails its checks stops the run with [`Error::InCall`],
    /// naming its place in [`Run::order`]; the calls before it are
    /// queued.
    pub fn execute(&self) -> Result<(), Error> {
        for (index, &id) in self.order.iter().enumerate() {
            self.execute_call(id).map_err(|error| Error::InCall {
                index,
                error: Box::new(error),
            })?;
        }
        Ok(())
    }

    /// Queues the call built from the spec with `id`, like one call of
    /// [`Bound::execute`].
    pub fn execute_call(&self, id: GlobalId<CallSpec>) -> Result<(), Error> {
        let (node, args) = self
            .calls
            .get(&id)
            .ok_or(Error::NotInPlan { what: "call" })?;
        self.scope.execute(*node, args)
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

impl<'d, B: Backend + Topology + Capturable> Run<'d, B> {
    /// Records one run, every call in order, to replay it. Nothing runs
    /// while recording. On a fail, gives the run back with the error.
    // The run comes back by value on a fail; a capture is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn capture(mut self) -> Result<CapturedRun<'d, B>, (Self, Error)> {
        let device = self.device;
        let recorded = device.scope(|s| {
            let bound = self.bind(s)?;
            s.capture(|| bound.execute())
        });
        match recorded {
            Ok(recording) => Ok(CapturedRun {
                run: self,
                recording,
            }),
            Err(e) => Err((self, e)),
        }
    }
}

/// A [`Run`] whose run is recorded once and replayed. Its tensors stay
/// where they were recorded: only their contents and the `Dynamic` meta
/// change between replays.
pub struct CapturedRun<'d, B: Backend + Topology + Capturable> {
    // Declared first, so the recording is released while its nodes and tensors still live.
    recording: Recording<'d, B>,
    run: Run<'d, B>,
}

impl<'d, B: Backend + Topology + Capturable> CapturedRun<'d, B> {
    /// [`Run::device`].
    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.run.device()
    }

    /// [`Run::tensors`].
    pub fn tensors(&self) -> impl Iterator<Item = (GlobalId<TensorSpec>, &AnyTensor<'d, B>)> {
        self.run.tensors()
    }

    /// [`Run::tensor`].
    pub fn tensor(&self, id: GlobalId<TensorSpec>) -> Result<&Tensor<'d, B>, Error> {
        self.run.tensor(id)
    }

    /// [`Run::calls`].
    pub fn calls(&self) -> impl Iterator<Item = (GlobalId<CallSpec>, &Call<'d, B>)> {
        self.run.calls()
    }

    /// [`Run::call`].
    pub fn call(&self, id: GlobalId<CallSpec>) -> Result<&Call<'d, B>, Error> {
        self.run.call(id)
    }

    /// [`Run::order`].
    pub fn order(&self) -> &[GlobalId<CallSpec>] {
        self.run.order()
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
        self.run
            .tensors
            .get_mut(&id)
            .ok_or(Error::NotInPlan { what: "tensor" })?
            .live_mut()
    }

    /// Drops the recording and gives back the [`Run`].
    pub fn release(self) -> Run<'d, B> {
        self.run
    }

    /// [`Run::bind`], for replays.
    pub fn bind<'scope, 'env, 's>(
        &'scope mut self,
        s: &'scope Scope<'scope, 'env, 's, B>,
    ) -> Result<CapturedBound<'scope, 'env, 's, 'd, B>, Error> {
        Ok(CapturedBound {
            bound: self.run.bind(s)?,
            recording: &self.recording,
        })
    }
}

/// A [`CapturedRun`] bound into a scope by [`CapturedRun::bind`].
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

#[cfg(test)]
#[path = "../../tests/run.rs"]
mod tests;
