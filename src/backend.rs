//! Everything a `backends/*` implementation must provide. Nothing else lives here.

use crate::core::device::DeviceId;
use crate::core::dtype::{DataKind, DataType, HostData};
use crate::core::node::{Binding, BuiltNode};
use crate::core::shader::{Shader, ShaderFormat, Workgroups};
use crate::core::table::BufferTable;

/// A backend's handle to one block of device memory.
///
/// Cloning a `Buffer` must not copy the memory: every clone points to the same
/// block, and the block lives until the last clone is dropped.
///
/// Core keeps a clone of each buffer a built node uses (see `BuiltNode`)
/// so a buffer can't be freed while that node can still run.
pub trait Buffer: Clone + Send + Sync + 'static {
    /// The real size of the memory block. It can be larger than the tensor
    /// stored in it; shaders take their bounds from the meta, not from this.
    fn size_bytes(&self) -> u64;

    /// How many clones of this buffer exist, this one included.
    /// 1 means nobody else holds it, so it can go back to the pool.
    fn holders(&self) -> usize;
}

/// One built dispatch, ready to run: a shader, its buffers, its meta and its
/// workgroup count, in the backend's own form.
///
/// Made by `Dispatch::build_node`
pub trait Node: Clone + Send + Sync + 'static {
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
}

/// How a backend finds devices on this machine and opens them.
pub trait Topology: Sized + Send + Sync + 'static {
    type Info: DeviceInfo;

    /// Every device this backend can open on this machine.
    fn choosable_devices() -> Vec<Self::Info>;

    /// Opens the device that `info` describes.
    fn attach(info: Self::Info) -> Result<Self, String>;

    /// The backend's name, like `"cuda"`.
    fn name(&self) -> &'static str;
}

/// The memory side of a device.
///
/// Uploads, downloads and dispatches run in the order they are called, so an
/// upload between two runs needs no `synchronize`. This holds for one stream
/// per device.
pub trait Storage: Send + Sync + 'static {
    type Buffer: Buffer;

    fn device_id(&self) -> DeviceId;

    /// This device's tensors
    fn table(&self) -> &BufferTable<Self::Buffer>;

    /// `DataKind` counterparts of `SupportsDType<D>`, for the mesh level where
    /// the dtype is only known at runtime. Each backend matches the kinds it
    /// has a `SupportsDType` impl for.
    /// a default body can't see which exist.
    fn alloc_kind(&self, kind: DataKind, elem_count: usize) -> Result<Self::Buffer, String>;

    /// Writes host data into `buf`. Fails if this backend doesn't support the
    /// data's kind.
    fn upload_kind(&self, buf: &Self::Buffer, data: &HostData) -> Result<(), String>;

    /// Reads `elem_count` elements of `kind` back from `buf` to the host.
    /// Blocks until the device work queued before it is done. Returns only
    /// those elements, not the rest of a larger buffer.
    fn download_kind(
        &self,
        buf: &Self::Buffer,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<HostData, String>;
}

/// The compute side of a device: builds nodes and runs them.
pub trait Dispatch: Storage + Sized {
    type Node: Node;

    /// The shader code format this backend runs. A graph is only built if
    /// every node's shader has code in this format.
    const FORMAT: ShaderFormat;

    /// Builds one node. `meta` is the words for slot 0; the backend makes the
    /// meta buffer and owns it.
    ///
    /// The graph checks the node before calling this, so a backend may
    /// panic on input the rules would have rejected.
    fn build_node(
        &self,
        shader: &'static Shader,
        meta: &[u32],
        bindings: &[(Binding, Self::Buffer)],
        workgroups: Workgroups,
    ) -> Self::Node;
    /// Writes new words into a node's meta buffer. Called before a run, for
    /// nodes with per-run meta fields. It must be ordered before that run's
    /// dispatches and must not change the meta buffer's address.
    fn update_meta(&self, node: &Self::Node, meta: &[u32]);

    /// Queues `nodes` to run in order. Doesn't wait for them to finish.
    fn execute(&self, nodes: &[BuiltNode<Self>]);

    /// Blocks until all queued work on this device is done.
    fn synchronize(&self);

    /// Like `execute`, but the backend may record `nodes` once under `key`
    /// and replay the recording on later calls (CUDA graphs, for example).
    /// The default just calls `execute`.
    ///
    /// A recording keeps raw buffer addresses, so the nodes' buffers must stay
    fn execute_captured(&self, _key: usize, nodes: &[BuiltNode<Self>]) {
        self.execute(nodes);
    }

    /// Drops the recording made under `key`, if there is one.
    fn release_captured(&self, _key: usize) {}

    /// True if this device can copy straight to `other`, without going
    /// through the host. Both must be devices of the same backend.
    fn supports_p2p(&self, other: DeviceId) -> bool {
        let _ = other;
        false
    }
}

/// A complete backend. Every `Dispatch` is one; code that takes a backend
/// asks for this.
pub trait Backend: Dispatch {}
impl<T: Dispatch> Backend for T {}

/// A backend implements this once for each data type it supports.
/// It is the typed form of `alloc_kind`, `upload_kind` and `download_kind`:
/// a wrong type is a compile error here instead of an `Err`.
pub trait SupportsDType<D: DataType>: Storage {
    /// Same contract as `Storage::alloc_kind`.
    fn alloc(&self, elem_count: usize) -> Self::Buffer;
    fn upload(&self, buf: &Self::Buffer, data: &[D::HostRepr]);
    fn download(&self, buf: &Self::Buffer, elem_count: usize) -> Vec<D::HostRepr>;

    /// A copy of `buf` in a new buffer on `dest`; `buf` stays.
    fn copy_to(&self, buf: &Self::Buffer, elem_count: usize, dest: &Self) -> Self::Buffer {
        let data = self.download(buf, elem_count);
        let copy = dest.alloc(elem_count);
        dest.upload(&copy, &data);
        copy
    }
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
}

/// Cutting tensors of type `D` out of an `Area`.
pub trait SupportsCarve<D: DataType>: Areable + SupportsDType<D> {
    /// Cuts a buffer for `elem_count` elements out of `area`.
    fn carve(&self, area: &mut Self::Area, elem_count: usize) -> Self::Buffer;
}

#[cfg(test)]
#[path = "tests/backend.rs"]
pub(crate) mod tests;
