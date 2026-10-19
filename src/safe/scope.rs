use std::cell::RefCell;
use std::marker::PhantomData;

use crate::backend::dtype::HostData;
use crate::backend::shader::BindingRole;
use crate::backend::{Access, Backend, Buffer as _, Capturable, Topology};

use super::{Buf, ComputeDevice, MetaValue, Node};

/// Invariant in 'scope, so a slot of one scope can't be given to another.
type Brand<'scope> = PhantomData<fn(&'scope ()) -> &'scope ()>;

/// 'env is what the closure borrows from outside, 'd the device. They are
/// apart so a `Recording` borrows only the device, not the buffers.
pub struct Scope<'scope, 'env: 'scope, 'd: 'env, B: Backend + Topology> {
    dev: &'d ComputeDevice<B>,
    /// Each pointer comes from a `&'scope mut` given to `bind`/`bind_node`,
    /// so it stays valid and unaliased until the scope ends.
    bufs: RefCell<Vec<*mut Buf<'d, B>>>,
    nodes: RefCell<Vec<*mut Node<'d, B>>>,
    /// `Some` while `capture` runs: what each `run` queued, by address.
    recording: RefCell<Option<Vec<RecordedCall>>>,
    _brand: Brand<'scope>,
    _env: PhantomData<&'env mut &'env ()>,
}

#[derive(Clone, Copy)]
pub struct BufSlot<'scope> {
    index: usize,
    _brand: Brand<'scope>,
}

#[derive(Clone, Copy)]
pub struct NodeSlot<'scope> {
    index: usize,
    _brand: Brand<'scope>,
}

#[derive(Clone, Copy)]
pub enum Arg<'scope> {
    Read(BufSlot<'scope>),
    Write(BufSlot<'scope>),
}

impl<'scope> Arg<'scope> {
    fn slot(self) -> BufSlot<'scope> {
        match self {
            Arg::Read(slot) | Arg::Write(slot) => slot,
        }
    }
}

impl<B: Backend + Topology> ComputeDevice<B> {
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
    pub fn bind<'b>(&self, buf: &'scope mut Buf<'b, B>) -> Result<BufSlot<'scope>, String> {
        if !std::ptr::eq(buf.dev, self.dev) {
            return Err("bind: the buffer belongs to another device".to_string());
        }
        let mut bufs = self.bufs.borrow_mut();
        bufs.push((buf as *mut Buf<'b, B>).cast());

        Ok(BufSlot {
            index: bufs.len() - 1,
            _brand: PhantomData,
        })
    }

    pub fn bind_node<'b>(&self, node: &'scope mut Node<'b, B>) -> Result<NodeSlot<'scope>, String> {
        if !std::ptr::eq(node.dev, self.dev) {
            return Err("bind_node: the node belongs to another device".to_string());
        }
        let mut nodes = self.nodes.borrow_mut();
        nodes.push((node as *mut Node<'b, B>).cast());

        Ok(NodeSlot {
            index: nodes.len() - 1,
            _brand: PhantomData,
        })
    }

    pub fn upload(&self, slot: BufSlot<'scope>, data: &HostData) -> Result<(), String> {
        let buf = self.bufs.borrow()[slot.index];
        // SAFETY: see `bufs`; no other reference to this buffer lives during the call.
        unsafe { &mut *buf }.upload(data)
    }

    pub fn download(&self, slot: BufSlot<'scope>) -> HostData {
        let buf = self.bufs.borrow()[slot.index];
        // SAFETY: see `bufs`.
        unsafe { &*buf }.download()
    }

    pub fn update_meta(&self, slot: NodeSlot<'scope>, meta: &[MetaValue]) -> Result<(), String> {
        let node = self.nodes.borrow()[slot.index];
        // SAFETY: see `nodes`; no other reference to this node lives during the call.
        unsafe { &mut *node }.update_meta(meta)
    }

    pub fn run(&self, node: NodeSlot<'scope>, args: &[Arg<'scope>]) -> Result<(), String> {
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
        unsafe { self.dev.raw.execute_raw(&node.raw, &mut access) }?;

        if let (Some(calls), Some(call)) = (self.recording.borrow_mut().as_mut(), recorded) {
            calls.push(call);
        }
        Ok(())
    }

    // `run`'s checks, made again for every call a replay runs.
    fn check(&self, node: &Node<'d, B>, args: &[Arg<'scope>]) -> Result<(), String> {
        let bufs = self.bufs.borrow();
        let shader = node.shader();

        if args.len() != shader.layout.len() {
            return Err(format!(
                "shader `{}`: expected {} buffers, got {}",
                shader.name,
                shader.layout.len(),
                args.len()
            ));
        }
        for (i, (arg, role)) in args.iter().zip(shader.layout).enumerate() {
            let (BindingRole::Input(kind)
            | BindingRole::Output(kind)
            | BindingRole::InOut(kind)
            | BindingRole::Accumulate(kind)) = role;
            let wants_write = !matches!(role, BindingRole::Input(_));
            if matches!(arg, Arg::Write(_)) != wants_write {
                return Err(format!(
                    "shader `{}`: slot {} is {role:?}, it needs Arg::{}",
                    shader.name,
                    i + 1,
                    if wants_write { "Write" } else { "Read" }
                ));
            }
            // SAFETY: see `bufs`.
            let buf = unsafe { &*bufs[arg.slot().index] };
            if buf.kind != *kind {
                return Err(format!(
                    "shader `{}`: slot {} holds {kind:?}, the buffer holds {:?}",
                    shader.name,
                    i + 1,
                    buf.kind
                ));
            }
            if buf.len < node.lens[i] {
                return Err(format!(
                    "shader `{}`: slot {} needs {} elements for this meta, the buffer holds {}",
                    shader.name,
                    i + 1,
                    node.lens[i],
                    buf.len
                ));
            }
        }
        for (i, arg) in args.iter().enumerate() {
            let Arg::Write(written) = arg else { continue };
            let aliased = args
                .iter()
                .enumerate()
                .any(|(j, other)| j != i && other.slot().index == written.index);
            if aliased {
                return Err(format!(
                    "shader `{}`: slot {} is written, its buffer can't be in another slot",
                    shader.name,
                    i + 1
                ));
            }
        }
        Ok(())
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
    // Records the `run`s that `f` makes instead of running them.
    pub fn capture(
        &self,
        f: impl FnOnce() -> Result<(), String>,
    ) -> Result<Recording<'d, B>, String> {
        if self.recording.borrow().is_some() {
            return Err("capture: this scope is already capturing".to_string());
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

        let raw = self.dev.raw.end_capture()?;
        let recording = Recording {
            raw: Some(raw),
            calls,
            dev: self.dev,
        };
        // On `Err`, dropping `recording` releases what was recorded.
        result.map(|()| recording)
    }

    pub fn replay(&self, recording: &Recording<'_, B>) -> Result<(), String> {
        if !std::ptr::eq(recording.dev, self.dev) {
            return Err("replay: the recording belongs to another device".to_string());
        }
        if self.recording.borrow().is_some() {
            return Err("replay: this scope is capturing".to_string());
        }

        for call in &recording.calls {
            let index = self
                .nodes
                .borrow()
                .iter()
                .position(|&node| node as usize == call.node)
                .ok_or("replay: a recorded node isn't bound in this scope, or has moved")?;

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
                        .ok_or(format!(
                            "replay: shader `{}`: a recorded buffer isn't bound in this scope",
                            node.shader().name
                        ))?;
                    let slot = BufSlot {
                        index,
                        _brand: PhantomData,
                    };

                    Ok(if write {
                        Arg::Write(slot)
                    } else {
                        Arg::Read(slot)
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
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
    }
}
