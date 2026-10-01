//! Everything a `backends/*` implementation must provide. Nothing else lives here.

use crate::core::device::DeviceId;
use crate::core::dtype::{DataKind, DataType, HostData};
use crate::core::node::Binding;
use crate::core::shader::{Shader, Workgroups};
use crate::core::tensor::TensorId;

/// raw data capsule that implemented in Dispatch/Storage
pub trait Buffer: Clone + Send + Sync + 'static {
    fn size_bytes(&self) -> u64;
    fn is_sole_owner(&self) -> bool;
}

pub trait Node: Clone + Send + Sync + 'static {
    const MAX_WORKGROUPS_PER_DIM: u32 = u32::MAX;

    fn shader(&self) -> &'static Shader;
    fn workgroups(&self) -> Workgroups;

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

pub trait DeviceInfo: Clone + std::fmt::Debug + Send + Sync + 'static {
    fn label(&self) -> String;
    fn total_memory_bytes(&self) -> u64;
}

pub trait Topology: Sized + Send + Sync + 'static {
    type Info: DeviceInfo;

    fn choosable_devices() -> Vec<Self::Info>;
    fn attach(info: Self::Info) -> Result<Self, String>;
    fn name(&self) -> &'static str;
}

pub trait Storage: Send + Sync + 'static {
    type Buffer: Buffer;
    fn device_id(&self) -> DeviceId;
    fn contains(&self, id: TensorId) -> bool;
    fn drop_buffer(&self, id: TensorId);

    /// `DataKind` counterparts of `SupportsDType<D>`, for the mesh level where
    /// the dtype is only known at runtime. Each backend matches the kinds it
    /// has a `SupportsDType` impl for -- a default body can't see which exist.
    ///
    /// # Errors
    /// Must fail, not alias or overwrite, if `contains(id)` is already true
    /// `Tensor::with_id` relies on this to keep two live tensors from ever
    /// sharing one id's buffer.
    fn alloc_kind(&self, id: TensorId, kind: DataKind, elem_count: usize) -> Result<(), String>;
    fn upload_kind(&self, id: TensorId, data: &HostData) -> Result<(), String>;
    fn download_kind(&self, id: TensorId, kind: DataKind) -> Result<HostData, String>;
}

pub trait Dispatch: Storage {
    type Node: Node;

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding],
        workgroups: Workgroups,
    ) -> Self::Node;
    fn execute(&self, nodes: &[Self::Node]);
    fn synchronize(&self);

    fn execute_captured(&self, _key: usize, nodes: &[Self::Node]) {
        self.execute(nodes);
    }
    fn release_captured(&self, _key: usize) {}

    fn supports_p2p(&self, other: DeviceId) -> bool {
        let _ = other;
        false
    }
}

pub trait Backend: Dispatch {}
impl<T: Dispatch> Backend for T {}

/// backend implements it for each DType that backend supports it
/// gives type safety at raw
pub trait SupportsDType<D: DataType>: Storage {
    fn alloc(&self, id: TensorId, elem_count: usize);
    fn upload(&self, id: TensorId, data: &[D::HostRepr]);
    fn download(&self, id: TensorId) -> Vec<D::HostRepr>;

    /// Allocates the same id on `dest`; the copy on `self` stays.
    fn copy_to(&self, id: TensorId, dest: &Self) {
        let data = self.download(id);
        dest.alloc(id, data.len());
        dest.upload(id, &data);
    }
}

pub trait Area: Send + Sync + 'static {
    fn size_bytes(&self) -> u64;
    fn remaining_bytes(&self) -> u64;
}

pub trait Areable: Storage {
    type Area: Area;

    fn reserve(&self, total_bytes: u64) -> Self::Area;
    fn release(&self, area: Self::Area);
}

pub trait SupportsCarve<D: DataType>: Areable + SupportsDType<D> {
    fn carve(&self, area: &mut Self::Area, id: TensorId, elem_count: usize);
}

#[cfg(test)]
#[path = "tests/backend.rs"]
pub(crate) mod tests;
