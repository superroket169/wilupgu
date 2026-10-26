use std::cell::RefCell;
use std::marker::PhantomData;

use crate::backend::dtype::HostData;
use crate::backend::{Access, Backend, Buffer as _, Capturable, Topology};
use crate::error::Error;

use super::{copy_ranges, rules, ComputeDevice, MetaValue, Node, Tensor};

/// Invariant in 'scope, so a slot of one scope can't be given to another.
type Brand<'scope> = PhantomData<fn(&'scope ()) -> &'scope ()>;

/// GPU work on one device, made by [`ComputeDevice::scope`]. Buffers and nodes
/// are bound to it once, then named by their slots in as many runs as needed.
///
/// 'env is what the closure borrows from outside, 'd the device. They are
/// apart so a [`Recording`] borrows only the device, not the buffers.
pub struct Scope<'scope, 'env: 'scope, 'd: 'env, B: Backend + Topology> {
    dev: &'d ComputeDevice<B>,
    /// Each pointer comes from a `&'scope mut` given to `bind`/`bind_node`,
    /// so it stays valid and unaliased until the scope ends.
    bufs: RefCell<Vec<*mut Tensor<'d, B>>>,
    nodes: RefCell<Vec<*mut Node<'d, B>>>,
    /// `Some` while `capture` runs: what each `run` queued, by address.
    recording: RefCell<Option<Vec<RecordedCall>>>,
    _brand: Brand<'scope>,
    _env: PhantomData<&'env mut &'env ()>,
}

/// A buffer bound to a scope, by [`Scope::bind`]. It can't leave the scope
/// or be given to another one.
#[derive(Clone, Copy)]
pub struct TensorSlot<'scope> {
    index: usize,
    _brand: Brand<'scope>,
}

/// A node bound to a scope, by [`Scope::bind_node`]. It can't leave the scope
/// or be given to another one.
#[derive(Clone, Copy)]
pub struct NodeSlot<'scope> {
    index: usize,
    _brand: Brand<'scope>,
}

/// One tensor slot's buffer in [`Scope::run`].
#[derive(Clone, Copy)]
pub enum Arg<'scope> {
    /// For an `Input` slot.
    Read(TensorSlot<'scope>),
    /// For an `Output`, `InOut` or `Accumulate` slot.
    Write(TensorSlot<'scope>),
}

impl<'scope> Arg<'scope> {
    fn slot(self) -> TensorSlot<'scope> {
        match self {
            Arg::Read(slot) | Arg::Write(slot) => slot,
        }
    }
}

impl<B: Backend + Topology> ComputeDevice<B> {
    /// Runs `f` with a new [`Scope`] on this device, then waits for the device:
    /// on return and on panic, so every buffer and node bound in it stays
    /// alive and untouched by the host while the device uses it. Buffers and
    /// nodes are made before the scope, since a bound one has to outlive it.
    pub fn scope<'d, 'env, R>(
        &'d self,
        f: impl for<'scope> FnOnce(&'scope Scope<'scope, 'env, 'd, B>) -> R,
    ) -> R
    where
        'd: 'env,
    {
        // Declared before `scope`, so it drops after it: on return and on panic.
        let _sync = SyncOnDrop(self);
        let scope = Scope {
            dev: self,
            bufs: RefCell::new(Vec::new()),
            nodes: RefCell::new(Vec::new()),
            recording: RefCell::new(None),
            _brand: PhantomData,
            _env: PhantomData,
        };
        f(&scope)
    }
}

impl<'scope, 'env, 'd, B: Backend + Topology> Scope<'scope, 'env, 'd, B> {
    /// Binds `buf` until the scope ends. Fails if it belongs to another device.
    pub fn bind<'b>(&self, buf: &'scope mut Tensor<'b, B>) -> Result<TensorSlot<'scope>, Error> {
        rules::same_device(self.dev, buf.dev, "tensor")?;
        let mut bufs = self.bufs.borrow_mut();
        bufs.push((buf as *mut Tensor<'b, B>).cast());

        Ok(TensorSlot {
            index: bufs.len() - 1,
            _brand: PhantomData,
        })
    }

    /// Binds `node` until the scope ends. Fails if it belongs to another device.
    pub fn bind_node<'b>(&self, node: &'scope mut Node<'b, B>) -> Result<NodeSlot<'scope>, Error> {
        rules::same_device(self.dev, node.dev, "node")?;
        let mut nodes = self.nodes.borrow_mut();
        nodes.push((node as *mut Node<'b, B>).cast());

        Ok(NodeSlot {
            index: nodes.len() - 1,
            _brand: PhantomData,
        })
    }

    /// [`Tensor::upload`] on a bound buffer. It lands before the runs queued after it.
    pub fn upload(&self, slot: TensorSlot<'scope>, data: &HostData) -> Result<(), Error> {
        let buf = self.bufs.borrow()[slot.index];
        // SAFETY: see `bufs`; no other reference to this buffer lives during the call.
        unsafe { &mut *buf }.upload(data)
    }

    /// [`Tensor::download`] on a bound buffer: waits for the runs queued before it.
    pub fn download(&self, slot: TensorSlot<'scope>) -> HostData {
        let buf = self.bufs.borrow()[slot.index];
        // SAFETY: see `bufs`.
        unsafe { &*buf }.download()
    }

    /// [`Tensor::copy_into`] between two bound tensors, without waiting: it
    /// lands before the runs queued after it. Not recorded by a capture; it
    /// happens right away. Also fails if `src` and `dst` are the same tensor.
    pub fn copy(
        &self,
        src: TensorSlot<'scope>,
        src_offset: usize,
        dst: TensorSlot<'scope>,
        dst_offset: usize,
        len: usize,
    ) -> Result<(), Error> {
        if src.index == dst.index {
            return Err(Error::SameTensor);
        }
        let bufs = self.bufs.borrow();
        // SAFETY: see `bufs`; the two slots are different tensors, so the `&`
        // and the `&mut` don't alias.
        let (from_ptr, to_ptr) = (bufs[src.index], bufs[dst.index]);
        let (from_tensor, to_tensor) = unsafe { (&*from_ptr, &mut *to_ptr) };
        let (from, to) = copy_ranges(from_tensor, src_offset, to_tensor, dst_offset, len)?;

        // SAFETY: `copy_ranges` checked both ranges; both tensors are bound
        // until the scope ends, and `SyncOnDrop` synchronizes before that.
        unsafe {
            self.dev.raw.copy_raw(
                from_tensor.raw(),
                from.start,
                to_tensor.raw_mut(),
                to.start,
                to.bytes,
            )
        }
        .map_err(Error::Backend)
    }

    /// [`Node::update_meta`] on a bound node; the runs queued after it use it.
    pub fn update_meta(&self, slot: NodeSlot<'scope>, meta: &[MetaValue]) -> Result<(), Error> {
        let node = self.nodes.borrow()[slot.index];
        // SAFETY: see `nodes`; no other reference to this node lives during the call.
        unsafe { &mut *node }.update_meta(meta)
    }

    /// Queues `node` to run on `args`, one per tensor slot in slot order.
    /// Doesn't wait for it to finish.
    ///
    /// Fails, before anything reaches the device, if the slot count is wrong,
    /// an `Input` slot doesn't get [`Arg::Read`] or another slot
    /// [`Arg::Write`], a buffer's kind isn't its slot's, a buffer is shorter
    /// than the shader's `min_lens` for the node's meta, or a written buffer is
    /// in another slot too.
    pub fn run(&self, node: NodeSlot<'scope>, args: &[Arg<'scope>]) -> Result<(), Error> {
        // SAFETY: see `nodes`.
        let node = unsafe { &*self.nodes.borrow()[node.index] };
        self.check(node, args)?;
        let bufs = self.bufs.borrow();

        let recorded = self.recording.borrow().is_some().then(|| RecordedCall {
            node: std::ptr::from_ref(node) as usize,
            args: args
                .iter()
                // SAFETY: see `bufs`.
                .map(|arg| {
                    let address = unsafe { &*bufs[arg.slot().index] }.raw().address();
                    match arg {
                        Arg::Read(_) => RecordedArg::Read(address),
                        Arg::Write(_) => RecordedArg::Write(address),
                    }
                })
                .collect(),
        });

        // SAFETY (each pointer): see `bufs`. A written buffer is in no other
        // slot (`check`), so its `&mut` is unique.
        let mut access: Vec<Access<'_, B::Buffer>> = args
            .iter()
            .map(|arg| match arg {
                Arg::Read(slot) => Access::Read(unsafe { &*bufs[slot.index] }.raw()),
                Arg::Write(slot) => {
                    let buf = bufs[slot.index];
                    Access::Write(unsafe { &mut *buf }.raw_mut())
                }
            })
            .collect();

        // SAFETY:
        // - slot count, order, roles and kinds are checked by `check`;
        //   `ComputeDevice::node` checked that the device supports every kind in the layout.
        // - every buffer holds at least the shader's `min_lens` for the node's meta
        //   (`check`); the shader's code stays within them (docs/SHADERS.md).
        // - buffers and node are borrowed until the scope ends, and `SyncOnDrop`
        //   synchronizes before that, so they live and the host can't touch them.
        unsafe { self.dev.raw.execute_raw(&node.raw, &mut access) }.map_err(Error::Backend)?;

        if let (Some(calls), Some(call)) = (self.recording.borrow_mut().as_mut(), recorded) {
            calls.push(call);
        }
        Ok(())
    }

    // `run`'s checks, made again for every call a replay runs.
    fn check(&self, node: &Node<'d, B>, args: &[Arg<'scope>]) -> Result<(), Error> {
        let bufs = self.bufs.borrow();
        let shader = node.shader();
        // SAFETY: see `bufs`.
        let tensors: Vec<&Tensor<'d, B>> = args
            .iter()
            .map(|arg| unsafe { &*bufs[arg.slot().index] })
            .collect();

        rules::slot_kinds(shader, tensors.iter().map(|t| t.kind))?;
        for (slot, (arg, role)) in args.iter().zip(shader.layout).enumerate() {
            rules::slot_access(shader, slot, role, matches!(arg, Arg::Write(_)))?;
            rules::long_enough(shader, slot, node.lens[slot], tensors[slot].len)?;
        }
        let indices: Vec<usize> = args.iter().map(|arg| arg.slot().index).collect();
        rules::not_aliased(shader, &indices)
    }
}

// Not a `Drop` on `Scope` itself: the closure borrows `Scope` for its own
// invariant 'scope, and a `Drop` there would make that borrow outlive it.
struct SyncOnDrop<'env, B: Backend + Topology>(&'env ComputeDevice<B>);

impl<B: Backend + Topology> Drop for SyncOnDrop<'_, B> {
    // Runs on panic too, so borrowed buffers are never released while the device uses them.
    fn drop(&mut self) {
        self.0.raw.synchronize();
    }
}

/// One `run` queued during `capture`: the node and each slot's buffer, by address.
struct RecordedCall {
    node: usize,
    args: Vec<RecordedArg>,
}

enum RecordedArg {
    Read(u64),
    Write(u64),
}

/// Runs recorded by [`Scope::capture`], to run again with [`Scope::replay`].
/// It borrows only the device; dropping it frees what the device holds for it.
pub struct Recording<'d, B: Backend + Topology + Capturable> {
    // `Some` until `Drop` releases it.
    raw: Option<B::Capture>,
    calls: Vec<RecordedCall>,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology + Capturable> Drop for Recording<'_, B> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            self.dev.raw.release_capture(raw);
        }
    }
}

/// Ends a capture that `f` left by panicking, so the device doesn't stay recording.
struct EndCaptureOnDrop<'a, B: Backend + Topology + Capturable> {
    dev: &'a ComputeDevice<B>,
    recording: &'a RefCell<Option<Vec<RecordedCall>>>,
    armed: bool,
}

impl<B: Backend + Topology + Capturable> Drop for EndCaptureOnDrop<'_, B> {
    fn drop(&mut self) {
        if self.armed {
            self.recording.borrow_mut().take();
            if let Ok(raw) = self.dev.raw.end_capture() {
                self.dev.raw.release_capture(raw);
            }
        }
    }
}

impl<'scope, 'env, 'd, B: Backend + Topology + Capturable> Scope<'scope, 'env, 'd, B> {
    /// Records the [`Scope::run`]s that `f` makes instead of running them;
    /// each one is checked as usual. Uploads and downloads inside `f` aren't
    /// recorded: they happen right away.
    ///
    /// Fails if this scope is already capturing, or if `f` fails; then nothing
    /// is kept, also when `f` panics.
    pub fn capture(
        &self,
        f: impl FnOnce() -> Result<(), Error>,
    ) -> Result<Recording<'d, B>, Error> {
        if self.recording.borrow().is_some() {
            return Err(Error::AlreadyCapturing);
        }
        *self.recording.borrow_mut() = Some(Vec::new());
        self.dev.raw.begin_capture();
        let mut guard = EndCaptureOnDrop {
            dev: self.dev,
            recording: &self.recording,
            armed: true,
        };
        let result = f();
        guard.armed = false;

        let calls = self
            .recording
            .borrow_mut()
            .take()
            .expect("set above, only taken here or on panic");

        let raw = self.dev.raw.end_capture().map_err(Error::Backend)?;
        let recording = Recording {
            raw: Some(raw),
            calls,
            dev: self.dev,
        };
        // On `Err`, dropping `recording` releases what was recorded.
        result.map(|()| recording)
    }

    /// Queues the recorded runs again, in order, each node with its current
    /// meta. Doesn't wait for them to finish.
    ///
    /// Every recorded buffer and node must be bound in this scope, in any
    /// order; a node that has moved since the recording counts as another node.
    /// Every run's checks are made again first, so a fail leaves the buffers
    /// untouched. Also fails for another device's recording, or inside
    /// [`Scope::capture`].
    pub fn replay(&self, recording: &Recording<'_, B>) -> Result<(), Error> {
        rules::same_device(self.dev, recording.dev, "recording")?;
        if self.recording.borrow().is_some() {
            return Err(Error::ReplayWhileCapturing);
        }

        for call in &recording.calls {
            let index = self
                .nodes
                .borrow()
                .iter()
                .position(|&node| node as usize == call.node)
                .ok_or(Error::NotBound { what: "node" })?;

            // SAFETY: see `nodes`.
            let node = unsafe { &*self.nodes.borrow()[index] };
            let args = call
                .args
                .iter()
                .map(|arg| {
                    let (address, write) = match *arg {
                        RecordedArg::Read(address) => (address, false),
                        RecordedArg::Write(address) => (address, true),
                    };
                    // Two bound buffers share an address only if both are empty,
                    // so a wrong pick here can only make `check` fail.
                    let index = self
                        .bufs
                        .borrow()
                        .iter()
                        // SAFETY: see `bufs`.
                        .position(|&buf| unsafe { &*buf }.raw().address() == address)
                        .ok_or(Error::NotBound { what: "tensor" })?;
                    let slot = TensorSlot {
                        index,
                        _brand: PhantomData,
                    };

                    Ok(if write {
                        Arg::Write(slot)
                    } else {
                        Arg::Read(slot)
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            self.check(node, &args)?;
        }

        // SAFETY: every recorded call passed `check` again against buffers and
        // nodes bound in this scope, at the recorded addresses; so they live,
        // haven't moved, and stay untouched by the host until `SyncOnDrop`.
        unsafe {
            self.dev
                .raw
                .replay(recording.raw.as_ref().expect("only taken in drop"))
        }
        .map_err(Error::Backend)
    }
}
