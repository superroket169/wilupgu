use super::*;
use crate::id::GlobalId;
use crate::mesh::Parallelity;
use crate::resolver::Resolvable;
use std::sync::{Arc as StdArc, Mutex};

#[derive(Clone)]
struct ToyBuffer(StdArc<Mutex<Vec<u8>>>, DeviceId);
impl ToyBuffer {
    fn new(owner: DeviceId, bytes: usize) -> Self {
        Self(StdArc::new(Mutex::new(vec![0u8; bytes])), owner)
    }
}
impl Buffer for ToyBuffer {
    fn size_bytes(&self) -> u64 {
        self.0.lock().unwrap().len() as u64
    }
    fn is_sole_owner(&self) -> bool {
        StdArc::strong_count(&self.0) == 1
    }
    fn id(&self) -> BufferId {
        BufferId(StdArc::as_ptr(&self.0) as u64)
    }
    fn owner(&self) -> DeviceId {
        self.1
    }
}

static TOY_SHADER: Shader = Shader {
    name: "Toy",
    layout: &[],
    dispatch: &[],
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

struct ToyBackend(DeviceId);
impl ToyBackend {
    fn new() -> Self {
        Self(DeviceId::new())
    }
}
impl Storage for ToyBackend {
    type Buffer = ToyBuffer;
    fn device_id(&self) -> DeviceId {
        self.0
    }
    fn drop_buffer(&self, _buf: Self::Buffer) {}
}
impl Dispatch for ToyBackend {
    type Node = ToyNode;
    fn build_node(
        &self,
        _s: &'static Shader,
        _b: &[Binding<Self::Buffer>],
        _wg: Workgroups,
    ) -> Self::Node {
        ToyNode
    }
    fn execute(&self, _nodes: &[Self::Node]) {}
    fn synchronize(&self) {}
}

// ToyBackend never overrides execute_captured/release_captured --
// exercises Dispatch's default (runtime-fallback) bodies.

// ToyBackend only ever implements SupportsDType<F32> -- on purpose.
impl SupportsDType<F32> for ToyBackend {
    fn alloc(&self, elem_count: usize) -> Self::Buffer {
        ToyBuffer::new(self.device_id(), elem_count * 4)
    }
    fn upload(&self, buf: &Self::Buffer, data: &[f32]) {
        *buf.0.lock().unwrap() = bytemuck::cast_slice(data).to_vec();
    }
    fn download(&self, buf: &Self::Buffer) -> Vec<f32> {
        bytemuck::cast_slice(&buf.0.lock().unwrap()).to_vec()
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
    fn carve(&self, _area: &mut Self::Area, elem_count: usize) -> Self::Buffer {
        ToyBuffer::new(self.device_id(), elem_count * 4)
    }
}

#[test]
fn carve_from_area() {
    let ctx = StdArc::new(ToyBackend::new());
    let mut area = ctx.reserve(1024);
    let buf = ctx.carve(&mut area, 4);
    assert_eq!(buf.size_bytes(), 16);
    ctx.release(area);
}

static COPY_SHADER: Shader = Shader {
    name: "Copy",
    layout: &[
        BindingRole::Input(DataKind::F32),
        BindingRole::Output(DataKind::F32),
    ],
    dispatch: &[],
};

static META_SHADER: Shader = Shader {
    name: "MetaEcho",
    layout: &[BindingRole::Meta {
        fields: &[MetaField::Uint],
        kind: MetaKind::Static, // a shader's declared kind is irrelevant to `accepts`
    }],
    dispatch: &[],
};

#[test]
fn maximized_meta_is_not_flagged_dynamic() {
    let m = ToyBuffer::new(DeviceId::new(), 4);
    let bindings = [Binding::new(
        0,
        &m,
        BindingRole::Meta {
            fields: &[MetaField::Uint],
            kind: MetaKind::Maximized(Resolvable::new()),
        },
    )];
    let spec = NodeSpec {
        id: GlobalId::new(),
        shader: &META_SHADER,
        bindings: &bindings,
        workgroups: Workgroups::linear(1),
        parallelity: Parallelity::Data,
    };
    let has_dynamic_meta = validate_spec::<ToyNode, _>(&spec).unwrap();
    assert!(
        !has_dynamic_meta,
        "Maximized must not be treated as Dynamic"
    );
}

#[test]
fn graph_build_and_run() {
    let ctx = StdArc::new(ToyBackend::new());
    let a = ToyBuffer::new(ctx.device_id(), 4);
    let b = ToyBuffer::new(ctx.device_id(), 4);
    let bindings = [
        Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
        Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
    ];
    let specs = [NodeSpec {
        id: GlobalId::new(),
        shader: &COPY_SHADER,
        bindings: &bindings,
        workgroups: Workgroups::linear(1),
        parallelity: Parallelity::Data,
    }];
    let graph = Graph::build(ctx, &specs).unwrap();
    graph.run();
}

#[test]
fn graph_capture_falls_back_to_execute() {
    let ctx = StdArc::new(ToyBackend::new());
    let a = ToyBuffer::new(ctx.device_id(), 4);
    let b = ToyBuffer::new(ctx.device_id(), 4);
    let bindings = [
        Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
        Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
    ];
    let specs = [NodeSpec {
        id: GlobalId::new(),
        shader: &COPY_SHADER,
        bindings: &bindings,
        workgroups: Workgroups::linear(1),
        parallelity: Parallelity::Data,
    }];
    let mut graph = Graph::build(ctx, &specs).unwrap();
    graph.capture(7, 0..1);
    assert!(matches!(
        graph.plan.as_slice(),
        [DispatchPlan::Captured { key: 7, .. }]
    ));
    graph.run(); // ToyBackend has no real capture support, runs via the default fallback
}

#[test]
fn graph_build_rejects_unordered_double_write() {
    let ctx = StdArc::new(ToyBackend::new());
    let a = ToyBuffer::new(ctx.device_id(), 4);
    let b = ToyBuffer::new(ctx.device_id(), 4);
    let bindings1 = [
        Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
        Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
    ];
    let bindings2 = [
        Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
        Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
    ];
    let specs = [
        NodeSpec {
            id: GlobalId::new(),
            shader: &COPY_SHADER,
            bindings: &bindings1,
            workgroups: Workgroups::linear(1),
            parallelity: Parallelity::Data,
        },
        NodeSpec {
            id: GlobalId::new(),
            shader: &COPY_SHADER,
            bindings: &bindings2,
            workgroups: Workgroups::linear(1),
            parallelity: Parallelity::Data,
        },
    ];
    let err = Graph::build(ctx, &specs).err().unwrap();
    assert!(err.contains("Buffer hazard"), "unexpected error: {err}");
}

#[test]
fn graph_build_rejects_foreign_buffer() {
    let ctx = StdArc::new(ToyBackend::new());
    let a = ToyBuffer::new(ctx.device_id(), 4);
    let b = ToyBuffer::new(DeviceId::new(), 4); // allocated on a different "device"
    let bindings = [
        Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
        Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
    ];
    let specs = [NodeSpec {
        id: GlobalId::new(),
        shader: &COPY_SHADER,
        bindings: &bindings,
        workgroups: Workgroups::linear(1),
        parallelity: Parallelity::Data,
    }];
    let err = Graph::build(ctx, &specs).err().unwrap();
    assert!(
        err.contains("Buffer ownership mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn global_id_is_unique() {
    let a = GlobalId::new();
    let b = GlobalId::new();
    assert_ne!(a, b);
}

#[test]
fn tensor_size_variants_construct() {
    let fixed = TensorSize::Fixed(128);
    let resolvable = TensorSize::Resolvable {
        size: Resolvable::new(),
        multiplier: 64,
        coefficient: 1,
    };
    assert!(matches!(fixed, TensorSize::Fixed(128)));
    assert!(matches!(resolvable, TensorSize::Resolvable { .. }));
}

#[test]
fn tensor_size_resolvable_groups_by_shared_clone() {
    let size = Resolvable::new();
    let a = TensorSize::Resolvable {
        size: size.clone(),
        multiplier: 64,
        coefficient: 1,
    };
    let b = TensorSize::Resolvable {
        size: size.clone(),
        multiplier: 64,
        coefficient: 1,
    };
    size.resolver().resolve(256);
    let TensorSize::Resolvable { size: a_size, .. } = a else {
        unreachable!()
    };
    let TensorSize::Resolvable { size: b_size, .. } = b else {
        unreachable!()
    };
    assert_eq!(*a_size.value(), 256);
    assert_eq!(*b_size.value(), 256);
}

#[test]
fn tensor_spec_blank_has_no_init() {
    let spec = TensorSpec::<F32>::blank(TensorSize::Fixed(4));
    assert!(spec.init().is_none());
}

#[test]
fn tensor_spec_seeded_carries_its_init() {
    let spec = TensorSpec::<F32>::seeded(
        TensorSize::Fixed(4),
        InitRecipe::UploadFromHost(vec![1.0, 2.0, 3.0, 4.0]),
    );
    assert!(matches!(
        spec.init(),
        Some(InitRecipe::UploadFromHost(data)) if data.len() == 4
    ));
}

#[test]
fn tensor_spec_ids_are_distinct() {
    let a = TensorSpec::<F32>::blank(TensorSize::Fixed(4));
    let b = TensorSpec::<F32>::blank(TensorSize::Fixed(4));
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
    let buf = src.alloc(4);
    src.upload(&buf, &[1.0, 2.0, 3.0, 4.0]);

    let copied = src.copy_to(&buf, &dest);

    assert_eq!(copied.owner(), dest.device_id());
    assert_eq!(dest.download(&copied), vec![1.0, 2.0, 3.0, 4.0]);
}
