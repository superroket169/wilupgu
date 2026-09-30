//! Everything a `backends/*` implementation must provide. Nothing else lives here.

use crate::device::DeviceId;
use crate::dtype::{DataKind, DataType, HostData};
use crate::node::Binding;
use crate::shader::{Shader, Workgroups};
use crate::tensor::TensorId;

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
pub(crate) mod tests {
    use super::*;
    use crate::dtype::F32;
    use crate::node::NodeSpec;
    use crate::shader::BindingRole;
    use std::collections::HashMap;
    use std::sync::{Arc as StdArc, Mutex};

    #[derive(Clone)]
    pub(crate) struct ToyBuffer(pub(crate) StdArc<Mutex<Vec<u8>>>);
    impl Buffer for ToyBuffer {
        fn size_bytes(&self) -> u64 {
            self.0.lock().unwrap().len() as u64
        }
        fn is_sole_owner(&self) -> bool {
            StdArc::strong_count(&self.0) == 1
        }
    }

    pub(crate) static TOY_SHADER: Shader = Shader {
        name: "Toy",
        layout: &[],
        shader_code: &[],
    };

    pub(crate) static COPY_SHADER: Shader = Shader {
        name: "Copy",
        layout: &[
            BindingRole::Input(DataKind::F32),
            BindingRole::Output(DataKind::F32),
        ],
        shader_code: &[],
    };

    pub(crate) static META_SHADER: Shader = Shader {
        name: "MetaEcho",
        layout: &[BindingRole::Meta {
            fields: &[crate::shader::MetaField::Uint],
            kind: crate::shader::MetaKind::Static, // a shader's declared kind is irrelevant to `accepts`
        }],
        shader_code: &[],
    };

    #[derive(Clone)]
    pub(crate) struct ToyNode;
    impl Node for ToyNode {
        fn shader(&self) -> &'static Shader {
            &TOY_SHADER
        }
        fn workgroups(&self) -> Workgroups {
            Workgroups::linear(1)
        }
    }

    pub(crate) struct ToyBackend {
        id: DeviceId,
        table: Mutex<HashMap<TensorId, ToyBuffer>>,
    }
    impl ToyBackend {
        pub(crate) fn new() -> Self {
            Self {
                id: DeviceId::new(),
                table: Mutex::new(HashMap::new()),
            }
        }
        fn insert(&self, id: TensorId, bytes: usize) {
            let buf = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; bytes])));
            self.table.lock().unwrap().insert(id, buf);
        }
        fn get(&self, id: TensorId) -> ToyBuffer {
            self.table
                .lock()
                .unwrap()
                .get(&id)
                .cloned()
                .expect("tensor not on this device")
        }
    }
    impl Storage for ToyBackend {
        type Buffer = ToyBuffer;
        fn device_id(&self) -> DeviceId {
            self.id
        }
        fn contains(&self, id: TensorId) -> bool {
            self.table.lock().unwrap().contains_key(&id)
        }
        fn drop_buffer(&self, id: TensorId) {
            self.table.lock().unwrap().remove(&id);
        }

        fn alloc_kind(
            &self,
            id: TensorId,
            kind: DataKind,
            elem_count: usize,
        ) -> Result<(), String> {
            match kind {
                DataKind::F32 => Ok(SupportsDType::<F32>::alloc(self, id, elem_count)),
                _ => Err(format!("toy does not support {kind:?}")),
            }
        }
        fn upload_kind(&self, id: TensorId, data: &HostData) -> Result<(), String> {
            match data {
                HostData::F32(v) => Ok(SupportsDType::<F32>::upload(self, id, v)),
                _ => Err(format!("toy does not support {:?}", data.kind())),
            }
        }
        fn download_kind(&self, id: TensorId, kind: DataKind) -> Result<HostData, String> {
            match kind {
                DataKind::F32 => Ok(F32::wrap(SupportsDType::<F32>::download(self, id))),
                _ => Err(format!("toy does not support {kind:?}")),
            }
        }
    }
    impl Dispatch for ToyBackend {
        type Node = ToyNode;
        fn build_node(&self, _s: &'static Shader, _b: &[Binding], _wg: Workgroups) -> Self::Node {
            ToyNode
        }
        fn execute(&self, _nodes: &[Self::Node]) {}
        fn synchronize(&self) {}
    }

    // ToyBackend never overrides execute_captured/release_captured --
    // exercises Dispatch's default (runtime-fallback) bodies.

    // ToyBackend only ever implements SupportsDType<F32> -- on purpose.
    impl SupportsDType<F32> for ToyBackend {
        fn alloc(&self, id: TensorId, elem_count: usize) {
            self.insert(id, elem_count * 4);
        }
        fn upload(&self, id: TensorId, data: &[f32]) {
            *self.get(id).0.lock().unwrap() = bytemuck::cast_slice(data).to_vec();
        }
        fn download(&self, id: TensorId) -> Vec<f32> {
            bytemuck::cast_slice(&self.get(id).0.lock().unwrap()).to_vec()
        }
    }

    #[derive(Clone, Debug)]
    pub(crate) struct ToyDevice(u32);

    impl DeviceInfo for ToyDevice {
        fn label(&self) -> String {
            format!("toy-device-{}", self.0)
        }
        fn total_memory_bytes(&self) -> u64 {
            1024
        }
    }

    impl Topology for ToyBackend {
        type Info = ToyDevice;
        fn choosable_devices() -> Vec<Self::Info> {
            vec![ToyDevice(0), ToyDevice(1)]
        }
        fn attach(_info: Self::Info) -> Result<Self, String> {
            Ok(ToyBackend::new())
        }
        fn name(&self) -> &'static str {
            "toy"
        }
    }

    #[test]
    fn topology_enumerate_then_attach() {
        let devices = ToyBackend::choosable_devices();
        assert_eq!(devices.len(), 2);
        let backend = ToyBackend::attach(devices[0].clone()).unwrap();
        assert_eq!(backend.name(), "toy");
    }

    pub(crate) struct ToyArea;

    impl Area for ToyArea {
        fn size_bytes(&self) -> u64 {
            1024
        }
        fn remaining_bytes(&self) -> u64 {
            1024
        }
    }

    impl Areable for ToyBackend {
        type Area = ToyArea;
        fn reserve(&self, _total_bytes: u64) -> Self::Area {
            ToyArea
        }
        fn release(&self, _area: Self::Area) {}
    }

    impl SupportsCarve<F32> for ToyBackend {
        fn carve(&self, _area: &mut Self::Area, id: TensorId, elem_count: usize) {
            self.insert(id, elem_count * 4);
        }
    }

    #[test]
    fn carve_from_area() {
        let ctx = ToyBackend::new();
        let id = TensorId::new();
        let mut area = ctx.reserve(1024);
        ctx.carve(&mut area, id, 4);
        assert_eq!(ctx.download(id).len(), 4);
        ctx.release(area);
    }

    pub(crate) fn copy_node(from: TensorId, to: TensorId) -> NodeSpec {
        NodeSpec::new(
            &COPY_SHADER,
            vec![
                Binding::new(0, from, BindingRole::Input(DataKind::F32)),
                Binding::new(1, to, BindingRole::Output(DataKind::F32)),
            ],
            Workgroups::linear(1),
        )
    }

    /// A device with `n` fresh one-element F32 tensors on it.
    pub(crate) fn device_with(n: usize) -> (StdArc<ToyBackend>, Vec<TensorId>) {
        let ctx = ToyBackend::new();
        let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();
        for &id in &ids {
            ctx.alloc(id, 1);
        }
        (StdArc::new(ctx), ids)
    }

    #[test]
    fn supports_p2p_defaults_to_false() {
        let ctx = ToyBackend::new();
        assert!(!ctx.supports_p2p(DeviceId::new()));
    }

    #[test]
    fn copy_to_round_trips_through_host() {
        let src = ToyBackend::new();
        let dest = ToyBackend::new();
        let id = TensorId::new();
        src.alloc(id, 4);
        src.upload(id, &[1.0, 2.0, 3.0, 4.0]);

        src.copy_to(id, &dest);

        assert_eq!(dest.download(id), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(src.contains(id), "the source copy stays");
    }

    #[test]
    fn drop_buffer_removes_it_from_the_table() {
        let ctx = ToyBackend::new();
        let id = TensorId::new();
        ctx.alloc(id, 1);
        ctx.drop_buffer(id);
        assert!(!ctx.contains(id));
    }

    #[test]
    fn alloc_kind_uses_the_matching_impl() {
        let ctx = ToyBackend::new();
        let id = TensorId::new();
        ctx.alloc_kind(id, DataKind::F32, 4).unwrap();
        assert!(ctx.contains(id));
        assert_eq!(ctx.download(id).len(), 4);
    }

    #[test]
    fn alloc_kind_rejects_an_unsupported_kind() {
        let err = ToyBackend::new()
            .alloc_kind(TensorId::new(), DataKind::Int4, 4)
            .unwrap_err();
        assert!(err.contains("Int4"), "{err}");
    }

    #[test]
    fn host_data_round_trips_through_the_kind_api() {
        let ctx = ToyBackend::new();
        let id = TensorId::new();
        ctx.alloc_kind(id, DataKind::F32, 3).unwrap();
        ctx.upload_kind(id, &HostData::F32(vec![1.0, 2.0, 3.0]))
            .unwrap();
        let back = ctx.download_kind(id, DataKind::F32).unwrap();
        assert!(matches!(back, HostData::F32(v) if v == vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn upload_kind_rejects_an_unsupported_kind() {
        let ctx = ToyBackend::new();
        let id = TensorId::new();
        ctx.alloc_kind(id, DataKind::F32, 1).unwrap();
        let err = ctx
            .upload_kind(id, &HostData::F16(vec![half::f16::ZERO]))
            .unwrap_err();
        assert!(err.contains("F16"), "{err}");
    }
}
