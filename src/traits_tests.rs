use super::*;
use crate::mesh::Parallelity;
use crate::resolver::Resolvable;
use crate::specs::*;
use std::collections::HashMap;
use std::sync::{Arc as StdArc, Mutex};

#[derive(Clone)]
struct ToyBuffer(StdArc<Mutex<Vec<u8>>>);
impl Buffer for ToyBuffer {
    fn size_bytes(&self) -> u64 {
        self.0.lock().unwrap().len() as u64
    }
    fn is_sole_owner(&self) -> bool {
        StdArc::strong_count(&self.0) == 1
    }
}

static TOY_SHADER: Shader = Shader {
    name: "Toy",
    layout: &[],
    shader_code: &[],
};

#[derive(Clone)]
struct ToyNode;
impl Node for ToyNode {
    fn shader(&self) -> &'static Shader {
        &TOY_SHADER
    }
    fn workgroups(&self) -> Workgroups {
        Workgroups::linear(1)
    }
}

struct ToyBackend {
    id: DeviceId,
    table: Mutex<HashMap<TensorId, ToyBuffer>>,
}
impl ToyBackend {
    fn new() -> Self {
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

    fn alloc_kind(&self, id: TensorId, kind: DataKind, elem_count: usize) -> Result<(), String> {
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
struct ToyDevice(u32);

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

struct ToyArea;

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

static COPY_SHADER: Shader = Shader {
    name: "Copy",
    layout: &[
        BindingRole::Input(DataKind::F32),
        BindingRole::Output(DataKind::F32),
    ],
    shader_code: &[],
};

static META_SHADER: Shader = Shader {
    name: "MetaEcho",
    layout: &[BindingRole::Meta {
        fields: &[MetaField::Uint],
        kind: MetaKind::Static, // a shader's declared kind is irrelevant to `accepts`
    }],
    shader_code: &[],
};

fn copy_node(from: TensorId, to: TensorId) -> NodeSpec {
    NodeSpec::new(
        &COPY_SHADER,
        vec![
            Binding::new(0, from, BindingRole::Input(DataKind::F32)),
            Binding::new(1, to, BindingRole::Output(DataKind::F32)),
        ],
        Workgroups::linear(1),
        Parallelity::Data,
    )
}

/// A device with `n` fresh one-element F32 tensors on it.
fn device_with(n: usize) -> (StdArc<ToyBackend>, Vec<TensorId>) {
    let ctx = ToyBackend::new();
    let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();
    for &id in &ids {
        ctx.alloc(id, 1);
    }
    (StdArc::new(ctx), ids)
}

#[test]
fn maximized_meta_is_not_flagged_dynamic() {
    let spec = NodeSpec::new(
        &META_SHADER,
        vec![Binding::new(
            0,
            TensorId::new(),
            BindingRole::Meta {
                fields: &[MetaField::Uint],
                kind: MetaKind::Maximized(ResolvedSize {
                    size: Resolvable::new(),
                    multiplier: 1,
                    coefficient: 0,
                }),
            },
        )],
        Workgroups::linear(1),
        Parallelity::Data,
    );
    let has_dynamic_meta = validate_spec::<ToyNode>(&spec).unwrap();
    assert!(
        !has_dynamic_meta,
        "Maximized must not be treated as Dynamic"
    );
}

#[test]
fn graph_build_and_run() {
    let (ctx, ids) = device_with(2);
    let graph = Graph::build(ctx, &[copy_node(ids[0], ids[1])]).unwrap();
    graph.run();
}

#[test]
fn graph_capture_falls_back_to_execute() {
    let (ctx, ids) = device_with(2);
    let mut graph = Graph::build(ctx, &[copy_node(ids[0], ids[1])]).unwrap();
    graph.capture(7, 0..1);
    assert!(matches!(
        graph.plan.as_slice(),
        [DispatchPlan::Captured { key: 7, .. }]
    ));
    graph.run(); // ToyBackend has no real capture support, runs via the default fallback
}

#[test]
fn graph_build_rejects_unordered_double_write() {
    let (ctx, ids) = device_with(2);
    let specs = [copy_node(ids[0], ids[1]), copy_node(ids[0], ids[1])];
    let err = Graph::build(ctx, &specs).err().unwrap();
    assert!(err.contains("Buffer hazard"), "unexpected error: {err}");
}

#[test]
fn graph_build_rejects_foreign_buffer() {
    let (ctx, ids) = device_with(1);
    let (_other, foreign) = device_with(1); // allocated on a different device
    let err = Graph::build(ctx, &[copy_node(ids[0], foreign[0])])
        .err()
        .unwrap();
    assert!(
        err.contains("Buffer ownership mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn global_id_is_unique() {
    let a = TensorId::new();
    let b = TensorId::new();
    assert_ne!(a, b);
}

#[test]
fn tensor_size_variants_construct() {
    let fixed = TensorSize::Fixed(128);
    let resolvable = TensorSize::Resolvable(ResolvedSize {
        size: Resolvable::new(),
        multiplier: 64,
        coefficient: 1,
    });
    assert!(matches!(fixed, TensorSize::Fixed(128)));
    assert!(matches!(resolvable, TensorSize::Resolvable(_)));
}

#[test]
fn tensor_size_resolvable_groups_by_shared_clone() {
    let size = Resolvable::new();
    let a = TensorSize::Resolvable(ResolvedSize {
        size: size.clone(),
        multiplier: 64,
        coefficient: 1,
    });
    let b = TensorSize::Resolvable(ResolvedSize {
        size: size.clone(),
        multiplier: 64,
        coefficient: 1,
    });
    size.resolver().resolve(256);
    let TensorSize::Resolvable(a_size) = a else {
        unreachable!()
    };
    let TensorSize::Resolvable(b_size) = b else {
        unreachable!()
    };
    assert_eq!(*a_size.size.value(), 256);
    assert_eq!(*b_size.size.value(), 256);
}

#[test]
fn tensor_spec_blank_has_no_init() {
    let spec = TensorSpec::blank::<F32>(TensorSize::Fixed(4));
    assert!(spec.init().is_none());
    assert_eq!(spec.kind(), DataKind::F32);
}

#[test]
fn tensor_spec_seeded_carries_its_init() {
    let spec = TensorSpec::seeded::<F32>(TensorSize::Fixed(4), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        spec.init(),
        Some(InitRecipe::UploadFromHost(HostData::F32(data))) if data.len() == 4
    ));
}

#[test]
fn tensor_spec_zeroed_keeps_its_kind() {
    let spec = TensorSpec::zeroed::<F16>(TensorSize::Fixed(4));
    assert!(matches!(spec.init(), Some(InitRecipe::Zero)));
    assert_eq!(spec.kind(), DataKind::F16);
}

#[test]
fn specs_of_different_kinds_share_one_slice() {
    let specs = [
        TensorSpec::blank::<F32>(TensorSize::Fixed(4)),
        TensorSpec::seeded::<Int8>(TensorSize::Fixed(2), vec![1, 2]),
    ];
    assert_eq!(specs[0].kind(), DataKind::F32);
    assert_eq!(specs[1].kind(), DataKind::Int8);
}

#[test]
fn host_data_kind_follows_its_variant() {
    assert_eq!(F16::wrap(vec![]).kind(), DataKind::F16);
    assert_eq!(Int4::wrap(vec![0]).kind(), DataKind::Int4);
}

#[test]
fn tensor_spec_ids_are_distinct() {
    let a = TensorSpec::blank::<F32>(TensorSize::Fixed(4));
    let b = TensorSpec::blank::<F32>(TensorSize::Fixed(4));
    assert_ne!(a.id(), b.id());
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

#[test]
fn unwrap_is_the_inverse_of_wrap() {
    assert_eq!(F32::unwrap(F32::wrap(vec![1.0, 2.0])), Some(vec![1.0, 2.0]));
    assert_eq!(F32::unwrap(F16::wrap(vec![])), None);
}
