//! Everything a `backends/*` implementation must provide,
//! and the backend tools (pool, io_log, ...)

pub mod dtype;
pub(crate) mod io_log;
pub(crate) mod pool;
pub mod shader;

use std::marker::PhantomData;

use crate::backend::dtype::DataKind;
use crate::backend::shader::{Shader, ShaderFormat, Workgroups};

/// A backend's handle to one block of device memory.
///
/// It has one owner: reading takes `&`, writing `&mut`. Dropping it frees
/// the block.
pub trait Buffer: Send + Sync + 'static {
    /// The real size of the memory block. It can be larger than the tensor
    /// stored in it; shaders take their bounds from the meta, not from this.
    fn size_bytes(&self) -> u64;

    /// Where the block starts, as the backend sees it: a device address on a
    /// GPU, a host pointer on the CPU. Tells whether a capture's buffers are
    /// still the same blocks.
    fn address(&self) -> u64;
}

/// One tensor slot's buffer, given to `execute_raw`:
/// `Read` for an `Input` slot
/// `Write` for `Output`, `InOut` and `Accumulate`.
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
    /// True if this device's shaders can read and write `kind`.
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

    /// The `DeviceInfo` this device was opened from.
    fn info(&self) -> &Self::Info;
}

/// The memory side of a device.
///
/// Uploads, downloads and dispatches run in the order they are called, so an
/// upload between two runs needs no `synchronize`. This holds for one stream
/// per device.
pub trait Storage: Send + Sync + 'static {
    type Buffer: Buffer;

    /// A new buffer of at least `bytes` bytes. Its contents are unspecified.
    fn alloc_raw(&self, bytes: u64) -> Result<Self::Buffer, String>;

    /// Writes `data` to the start of `buf`.
    ///
    /// # Safety
    /// `data.len()` must not be over `buf.size_bytes()`.
    unsafe fn upload_raw(&self, buf: &mut Self::Buffer, data: &[u8]);

    /// Reads the first `bytes` bytes of `buf` back to the host. Blocks until
    /// the device work queued before it is done.
    ///
    /// # Safety
    /// `bytes` must not be over `buf.size_bytes()`.
    unsafe fn download_raw(&self, buf: &Self::Buffer, bytes: u64) -> Vec<u8>;
}

/// The compute side of a device: builds nodes and runs them.
pub trait Dispatch: Storage {
    type Node: Node;

    /// The shader code format this backend runs. Only shaders with code in
    /// this format run on it.
    const FORMAT: ShaderFormat;

    /// Builds one node
    /// `meta` is the words for slot 0
    /// the backend makes the meta buffer and owns it.
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

    /// Queues `node` to run with `bindings`. Doesn't wait for it to finish.
    ///
    /// Fails if a binding's buffer belongs to another device.
    ///
    /// # Safety
    /// - `bindings` are in slot order (index 0 is slot 1), one for each entry
    ///   of the shader's layout.
    /// - An `Input` slot gets `Read`; every other slot gets `Write`.
    /// - Each buffer holds the kind its slot declares, and this device
    ///   supports that kind.
    /// - The node's meta describes sizes that fit the bound buffers.
    /// - Every buffer lives, and the host doesn't touch it, until
    ///   `synchronize` returns.
    unsafe fn execute_raw(
        &self,
        node: &Self::Node,
        bindings: &mut [Access<'_, Self::Buffer>],
    ) -> Result<(), String>;

    /// Blocks until all queued work on this device is done.
    fn synchronize(&self);

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

/// A backend that can record `execute_raw` calls once and replay them later
/// as one submission, like CUDA graphs.
pub trait Capturable: Dispatch {
    type Capture: Send + Sync + 'static;

    /// Starts recording. Until `end_capture`, `execute_raw` calls on this
    /// device are recorded instead of run.
    fn begin_capture(&self);

    /// Stops recording and returns what was recorded.
    fn end_capture(&self) -> Result<Self::Capture, String>;

    /// Queues the recorded calls to run again, in order. Doesn't wait for
    /// them to finish.
    ///
    /// # Safety
    /// - Every `execute_raw` safety rule holds for each recorded call.
    /// - Every recorded buffer still lives at the same address, and the host
    ///   doesn't touch it until `synchronize` returns.
    unsafe fn replay(&self, capture: &Self::Capture) -> Result<(), String>;

    /// Frees what the backend holds for `capture`.
    fn release_capture(&self, capture: Self::Capture);
}

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

    /// Cuts a buffer of `bytes` bytes out of `area`
    /// Fails if `area` has no room left
    /// The buffer borrows `area`, so `release` can't run while it lives.
    fn carve<'a>(
        &self,
        area: &'a Self::Area,
        bytes: u64,
    ) -> Result<Carved<'a, Self::Buffer>, String>;
}

/// A buffer cut out of an `Area`
/// It can't outlive the area
/// and its buffer is never handed out by value
/// so the borrow can't be escaped.
pub struct Carved<'a, Buf> {
    buf: Buf,
    _area: PhantomData<&'a ()>,
}

impl<Buf: Buffer> Carved<'_, Buf> {
    /// Wraps a buffer the backend cut out of an area.
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
