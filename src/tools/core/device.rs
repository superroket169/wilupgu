use crate::backend::dtype::{DataKind, HostData};
use crate::tools::core::id::{DeviceId, TensorId};
use crate::backend::{Dispatch, Storage};
#[cfg(test)]
use crate::backends::toy::ToyBackend;
#[cfg(feature = "cpu")]
use crate::backends::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backends::CudaBackend;
#[cfg(feature = "rayon")]
use crate::backends::RayonBackend;
use crate::tools::core::graph::Graph;
use crate::tools::core::node::NodeSpec;

#[cfg(not(any(feature = "cuda", feature = "cpu", feature = "rayon", test)))]
compile_error!("wilupgu needs at least one backend feature: `cuda`, `cpu` or `rayon`");

/// One live device, whatever its backend
/// It only dispatches on which backend it is
/// everything that depends on the dtype is matched inside the backend itself (`alloc_kind` & co).
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "cpu")]
    Cpu(CpuBackend),
    #[cfg(feature = "rayon")]
    Rayon(RayonBackend),
    #[cfg(test)]
    #[allow(private_interfaces)]
    Toy(ToyBackend),
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
            #[cfg(test)]
            Device::Toy($b) => $body,
        }
    };
}

impl Device {
    pub fn id(&self) -> DeviceId {
        on_backend!(self, b => b.device_id())
    }

    pub(crate) fn contains(&self, id: TensorId) -> bool {
        on_backend!(self, b => b.table().contains(id))
    }

    pub(crate) fn drop_buffer(&self, id: TensorId) {
        on_backend!(self, b => b.table().remove(id))
    }

    pub(crate) fn check_unused(&self, id: TensorId) -> Result<(), String> {
        on_backend!(self, b => crate::tools::core::rules::check_unused(b.table(), id, b.device_id()))
    }

    pub(crate) fn alloc_kind(
        &self,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<(), String> {
        on_backend!(self, b => b.table().alloc(b, id, kind, elem_count))
    }

    pub(crate) fn upload_kind(&self, id: TensorId, data: &HostData) -> Result<(), String> {
        on_backend!(self, b => b.table().upload(b, id, data))
    }

    pub(crate) fn download_kind(
        &self,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<HostData, String> {
        on_backend!(self, b => b.table().download(b, id, kind, elem_count))
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
            #[cfg(test)]
            (Device::Toy(a), Device::Toy(_)) => a.supports_p2p(other.id()),
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
            let graph = Graph::build(b, std::slice::from_ref(node))?;
            graph.run();
            b.synchronize();
            Ok(())
        })
    }
}
