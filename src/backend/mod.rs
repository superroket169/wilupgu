//! Everything a `backends/*` implementation must provide
//! and neceserry tools (pool & table & ...)

pub mod dtype;
pub mod id;
pub(crate) mod io_log;
pub(crate) mod pool;
pub mod shader;
pub mod table;

use std::marker::PhantomData;

use crate::backend::dtype::DataKind;
use crate::backend::shader::{Shader, ShaderFormat, Workgroups};

/// A backend's handle to one block of device memory.
pub trait Buffer: Send + Sync + 'static {
    /// The real size of the memory block. It can be larger than the tensor
    /// stored in it; shaders take their bounds from the meta, not from this.
    fn size_bytes(&self) -> u64;

    fn address(&self) -> u64;
}

pub enum Access<'a, Buf> {
    Read(&'a Buf),
    Write(&'a mut Buf),
}

/// One built dispatch, ready to run: a shader, its meta and its workgroup
/// count, in the backend's own form.
///
/// Made by `Dispatch::build_node`
pub trait Node: Send + Sync + 'static {
    /// The largest workgroup count this backend accepts in one dimension.
    const MAX_WORKGROUPS_PER_DIM: u32 = u32::MAX;

    fn shader(&self) -> &'static Shader;
    fn workgroups(&self) -> Workgroups;

    /// Fails if any dimension of `wg` is over `MAX_WORKGROUPS_PER_DIM`.
    fn validate_workgroups(wg: Workgroups) -> Result<(), String> {
        if wg.dims().iter().all(|&d| d <= Self::MAX_WORKGROUPS_PER_DIM) {
            Ok(())
        } else {
            Err(format!(
                "workgroup count {wg:?} exceeds this backend's limit of {} per dimension",
                Self::MAX_WORKGROUPS_PER_DIM
            ))
        }
    }
}

/// What a backend can tell about one device before it is opened.
pub trait DeviceInfo: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// A name for people to read, like the GPU's model name.
    fn label(&self) -> String;
    fn total_memory_bytes(&self) -> u64;
    fn supports_kind(&self, kind: DataKind) -> bool;
}

/// How a backend finds devices on this machine and opens them.
pub trait Topology: Sized + Send + Sync + 'static {
    type Info: DeviceInfo;

    /// The backend's name, like `"cuda"`.
    const NAME: &'static str;

    /// Every device this backend can open on this machine.
    fn choosable_devices() -> Vec<Self::Info>;

    /// Opens the device that `info` describes.
    fn attach(info: Self::Info) -> Result<Self, String>;

    fn info(&self) -> &Self::Info;
}

/// The memory side of a device.
///
/// Uploads, downloads and dispatches run in the order they are called, so an
/// upload between two runs needs no `synchronize`. This holds for one stream
/// per device.
pub trait Storage: Send + Sync + 'static {
    type Buffer: Buffer;

    fn alloc_raw(&self, bytes: u64) -> Result<Self::Buffer, String>;

    unsafe fn upload_raw(&self, buf: &mut Self::Buffer, data: &[u8]);

    /// Blocks until the device work queued before it is done.
    unsafe fn download_raw(&self, buf: &Self::Buffer, bytes: u64) -> Vec<u8>;
}

/// The compute side of a device: builds nodes and runs them.
pub trait Dispatch: Storage {
    type Node: Node;

    /// The shader code format this backend runs. A graph is only built if
    /// every node's shader has code in this format.
    const FORMAT: ShaderFormat;

    /// Builds one node. `meta` is the words for slot 0; the backend makes the
    /// meta buffer and owns it.
    fn build_node(
        &self,
        shader: &'static Shader,
        meta: &[u32],
        workgroups: Workgroups,
    ) -> Self::Node;

    /// Writes new words into a node's meta buffer. Called before a run, for
    /// nodes with per-run meta fields. It must be ordered before that run's
    /// dispatches and must not change the meta buffer's address.
    fn update_meta(&self, node: &mut Self::Node, meta: &[u32]);

    unsafe fn execute_raw(
        &self,
        node: &Self::Node,
        bindings: &mut [Access<'_, Self::Buffer>],
    ) -> Result<(), String>;

    /// Blocks until all queued work on this device is done.
    fn synchronize(&self);

    // TODO: capture methods, waiting on the capture shape decision.

    /// True if this device can copy straight to `other`, without going
    /// through the host. Both must be devices of the same backend.
    fn supports_p2p(&self, other: &Self) -> bool {
        let _ = other;
        false
    }
}

/// A complete backend. Every `Dispatch` is one; code that takes a backend
/// asks for this.
pub trait Backend: Dispatch {}
impl<T: Dispatch> Backend for T {}

/// One large block of device memory, reserved up front. Tensors are then
/// cut out of it instead of being allocated one by one.
pub trait Area: Send + Sync + 'static {
    fn size_bytes(&self) -> u64;

    /// Bytes not yet cut out.
    fn remaining_bytes(&self) -> u64;
}

/// A backend that can reserve `Area`s.
pub trait Areable: Storage {
    type Area: Area;

    fn reserve(&self, total_bytes: u64) -> Self::Area;

    /// Gives the whole area back to the device.
    fn release(&self, area: Self::Area);

    fn carve<'a>(
        &self,
        area: &'a Self::Area,
        bytes: u64,
    ) -> Result<Carved<'a, Self::Buffer>, String>;
}

pub struct Carved<'a, Buf> {
    buf: Buf,
    _area: PhantomData<&'a ()>,
}

impl<Buf: Buffer> Carved<'_, Buf> {
    pub fn new(buf: Buf) -> Self {
        Self {
            buf,
            _area: PhantomData,
        }
    }

    pub fn buffer(&self) -> &Buf {
        &self.buf
    }

    pub fn buffer_mut(&mut self) -> &mut Buf {
        &mut self.buf
    }
}

#[cfg(test)]
#[path = "../tests/backend.rs"]
pub(crate) mod tests;
