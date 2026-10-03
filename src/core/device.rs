use std::sync::Arc;

use crate::backend::{Dispatch, Storage};
#[cfg(feature = "cpu")]
use crate::backends::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backends::CudaBackend;
#[cfg(feature = "rayon")]
use crate::backends::RayonBackend;
use crate::core::dtype::{DataKind, HostData};
use crate::core::graph::Graph;
use crate::core::id::GlobalId;
use crate::core::node::NodeSpec;
use crate::core::tensor::TensorId;

/// Stand-in tag until each backend has its own concrete device-id source.
pub enum DeviceTag {}

pub type DeviceId = GlobalId<DeviceTag>;

#[cfg(not(any(feature = "cuda", feature = "cpu", feature = "rayon")))]
compile_error!("wilupgu needs at least one backend feature: `cuda`, `cpu` or `rayon`");

/// One live device, whatever its backend
/// It only dispatches on which backend it is
/// everything that depends on the dtype is matched inside the backend itself (`alloc_kind` & co).
#[derive(Clone)]
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaBackend>),
    #[cfg(feature = "cpu")]
    Cpu(Arc<CpuBackend>),
    #[cfg(feature = "rayon")]
    Rayon(Arc<RayonBackend>),
}

/// Runs `$body` with `$b` bound to the backend inside whichever variant this is.
macro_rules! on_backend {
    ($device:expr, $b:ident => $body:expr) => {
        match $device {
            #[cfg(feature = "cuda")]
            Device::Cuda($b) => $body,
            #[cfg(feature = "cpu")]
            Device::Cpu($b) => $body,
            #[cfg(feature = "rayon")]
            Device::Rayon($b) => $body,
        }
    };
}

impl Device {
    pub fn id(&self) -> DeviceId {
        on_backend!(self, b => b.device_id())
    }

    pub(crate) fn contains(&self, id: TensorId) -> bool {
        on_backend!(self, b => b.contains(id))
    }

    pub(crate) fn drop_buffer(&self, id: TensorId) {
        on_backend!(self, b => b.drop_buffer(id))
    }

    pub(crate) fn check_unused(&self, id: TensorId) -> Result<(), String> {
        on_backend!(self, b => crate::core::rules::check_unused(b.as_ref(), id))
    }

    pub(crate) fn alloc_kind(
        &self,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<(), String> {
        on_backend!(self, b => b.alloc_kind(id, kind, elem_count))
    }

    pub(crate) fn upload_kind(&self, id: TensorId, data: &HostData) -> Result<(), String> {
        on_backend!(self, b => b.upload_kind(id, data))
    }

    pub(crate) fn download_kind(&self, id: TensorId, kind: DataKind) -> Result<HostData, String> {
        on_backend!(self, b => b.download_kind(id, kind))
    }

    /// P2P only ever exists between two devices of the same backend.
    pub fn supports_p2p(&self, other: &Device) -> bool {
        match (self, other) {
            #[cfg(feature = "cuda")]
            (Device::Cuda(a), Device::Cuda(_)) => a.supports_p2p(other.id()),
            #[cfg(feature = "cpu")]
            (Device::Cpu(a), Device::Cpu(_)) => a.supports_p2p(other.id()),
            #[cfg(feature = "rayon")]
            (Device::Rayon(a), Device::Rayon(_)) => a.supports_p2p(other.id()),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Builds and runs a single node on this device
    /// blocking until done.
    /// For one-off work outside a mesh
    ///
    /// built-in `combine` ops
    /// and users' own ones.
    pub fn run_once(&self, node: &NodeSpec) -> Result<(), String> {
        on_backend!(self, b => {
            let graph = Graph::build(b.clone(), std::slice::from_ref(node))?;
            graph.run();
            b.synchronize();
            Ok(())
        })
    }
}
