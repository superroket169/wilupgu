use std::cell::RefCell;
use std::marker::PhantomData;

use crate::backend::dtype::HostData;
use crate::backend::shader::BindingRole;
use crate::backend::{Access, Backend, Topology};

use super::{Buf, ComputeDevice, MetaValue, Node};

// Invariant in 'scope, so a slot of one scope can't be given to another.
type Brand<'scope> = PhantomData<fn(&'scope ()) -> &'scope ()>;

pub struct Scope<'scope, 'env: 'scope, B: Backend + Topology> {
    dev: &'env ComputeDevice<B>,
    // Each pointer comes from a `&'scope mut` given to `bind`/`bind_node`,
    // so it stays valid and unaliased until the scope ends.
    bufs: RefCell<Vec<*mut Buf<'env, B>>>,
    nodes: RefCell<Vec<*mut Node<'env, B>>>,
    _brand: Brand<'scope>,
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
    pub fn scope<'env, R>(
        &'env self,
        f: impl for<'scope> FnOnce(&'scope Scope<'scope, 'env, B>) -> R,
    ) -> R {
        // Declared before `scope`, so it drops after it: on return and on panic.
        let _sync = SyncOnDrop(self);
        let scope = Scope {
            dev: self,
            bufs: RefCell::new(Vec::new()),
            nodes: RefCell::new(Vec::new()),
            _brand: PhantomData,
        };
        f(&scope)
    }
}

impl<'scope, 'env, B: Backend + Topology> Scope<'scope, 'env, B> {
    pub fn bind<'d>(&self, buf: &'scope mut Buf<'d, B>) -> Result<BufSlot<'scope>, String> {
        if !std::ptr::eq(buf.dev, self.dev) {
            return Err("bind: the buffer belongs to another device".to_string());
        }
        let mut bufs = self.bufs.borrow_mut();
        bufs.push((buf as *mut Buf<'d, B>).cast());

        Ok(BufSlot {
            index: bufs.len() - 1,
            _brand: PhantomData,
        })
    }

    pub fn bind_node<'d>(&self, node: &'scope mut Node<'d, B>) -> Result<NodeSlot<'scope>, String> {
        if !std::ptr::eq(node.dev, self.dev) {
            return Err("bind_node: the node belongs to another device".to_string());
        }
        let mut nodes = self.nodes.borrow_mut();
        nodes.push((node as *mut Node<'d, B>).cast());

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
        let bufs = self.bufs.borrow();
        // SAFETY: see `nodes`.
        let node = unsafe { &*self.nodes.borrow()[node.index] };
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

        // SAFETY (each pointer): see `bufs`. A written buffer is in no other
        // slot (checked above), so its `&mut` is unique.
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
        // - slot count, order, roles and kinds are checked above; `ComputeDevice::node`
        //   checked that the device supports every kind in the layout.
        // - every buffer holds at least the shader's `min_lens` for the node's meta
        //   (checked above); the shader's code stays within them (docs/SHADERS.md).
        // - buffers and node are borrowed until the scope ends, and `SyncOnDrop`
        //   synchronizes before that, so they live and the host can't touch them.
        unsafe { self.dev.raw.execute_raw(&node.raw, &mut access) }
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
