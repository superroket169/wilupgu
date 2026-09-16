use crate::backends::BackendDispatch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingRole {
    Input(DataKind),
    Output(DataKind),
    InOut(DataKind),
    Accumulate(DataKind),
    Meta { fields: MetaSlot, kind: MetaKind },
}

impl BindingRole {
    fn accepts(&self, actual: &BindingRole) -> bool {
        match (self, actual) {
            (BindingRole::Input(a), BindingRole::Input(b)) => a == b,
            (BindingRole::Output(a), BindingRole::Output(b)) => a == b,
            (BindingRole::InOut(a), BindingRole::InOut(b)) => a == b,
            (BindingRole::Accumulate(a), BindingRole::Accumulate(b)) => a == b,
            (BindingRole::Meta { .. }, BindingRole::Meta { .. }) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    F32,
    F16,
    Bf16,
    Int8,
    Int4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaKind {
    Static,
    Dynamic,
}

pub struct Binding<'a, Buf> {
    pub slot: u32,
    pub buffer: &'a Buf,
    pub mode: BindingRole,
}

impl<'a, Buf> Binding<'a, Buf> {
    #[inline]
    pub fn new(slot: u32, buffer: &'a Buf, mode: BindingRole) -> Self {
        Self { slot, buffer, mode }
    }
}

pub struct Shader {
    pub name: &'static str,
    pub layout: &'static [BindingRole],
    pub dispatch: &'static [BackendDispatch],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaField {
    Uint,
    Float,
}

pub type MetaSlot = &'static [MetaField];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroups {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Workgroups {
    pub const fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    fn dims(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

pub trait DataType: Copy + Send + Sync + 'static {
    type HostRepr: bytemuck::Pod + Default + Clone;
    const BITS_PER_ELEM: u32;
    const ELEMS_PER_HOST_WORD: u32 = 1;
    const KIND: DataKind;
}

#[derive(Clone, Copy)]
pub struct F32;
impl DataType for F32 {
    type HostRepr = f32;
    const BITS_PER_ELEM: u32 = 32;
    const KIND: DataKind = DataKind::F32;
}

#[derive(Clone, Copy)]
pub struct F16;
impl DataType for F16 {
    type HostRepr = half::f16;
    const BITS_PER_ELEM: u32 = 16;
    const KIND: DataKind = DataKind::F16;
}

#[derive(Clone, Copy)]
pub struct Bf16;
impl DataType for Bf16 {
    type HostRepr = half::bf16;
    const BITS_PER_ELEM: u32 = 16;
    const KIND: DataKind = DataKind::Bf16;
}

#[derive(Clone, Copy)]
pub struct Int8;
impl DataType for Int8 {
    type HostRepr = u8;
    const BITS_PER_ELEM: u32 = 8;
    const KIND: DataKind = DataKind::Int8;
}

#[derive(Clone, Copy)]
pub struct Int4;
impl DataType for Int4 {
    type HostRepr = u32;
    const BITS_PER_ELEM: u32 = 4;
    const ELEMS_PER_HOST_WORD: u32 = 8;
    const KIND: DataKind = DataKind::Int4;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferId(pub u64);

pub trait Buffer: Clone + Send + Sync + 'static {
    fn size_bytes(&self) -> u64;
    fn is_sole_owner(&self) -> bool;
    fn id(&self) -> BufferId;
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

pub trait Device: Clone + std::fmt::Debug + Send + Sync + 'static {
    fn label(&self) -> String;
    fn total_memory_bytes(&self) -> u64;
}

pub trait Topology: Sized + Send + Sync + 'static {
    type Device: Device;

    fn choosable_devices() -> Vec<Self::Device>;
    fn attach(device: Self::Device) -> Result<Self, String>;
    fn name(&self) -> &'static str;
}

pub trait Storage: Send + Sync + 'static {
    type Buffer: Buffer;
    fn drop_buffer(&self, buf: Self::Buffer);
}

pub trait Dispatch: Storage {
    type Node: Node;

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<Self::Buffer>],
        workgroups: Workgroups,
    ) -> Self::Node;
    fn execute(&self, nodes: &[Self::Node]);
    fn synchronize(&self);

    fn execute_captured(&self, _key: usize, nodes: &[Self::Node]) {
        self.execute(nodes);
    }
    fn release_captured(&self, _key: usize) {}
}

pub trait Backend: Dispatch {}
impl<T: Dispatch> Backend for T {}

pub trait SupportsDType<D: DataType>: Storage {
    fn alloc(&self, elem_count: usize) -> Self::Buffer;
    fn upload(&self, buf: &Self::Buffer, data: &[D::HostRepr]);
    fn download(&self, buf: &Self::Buffer) -> Vec<D::HostRepr>;
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
    fn carve(&self, area: &mut Self::Area, elem_count: usize) -> Self::Buffer;
}

pub struct NodeSpec<'a, Buf> {
    pub shader: &'static Shader,
    pub bindings: &'a [Binding<'a, Buf>],
    pub workgroups: Workgroups,
}

fn validate_spec<N: Node, Buf>(spec: &NodeSpec<Buf>) -> Result<bool, String> {
    N::validate_workgroups(spec.workgroups)
        .map_err(|e| format!("kernel `{}`: {e}", spec.shader.name))?;

    let layout = spec.shader.layout;
    let name = spec.shader.name;
    let mut covered = vec![false; layout.len()];
    let mut has_dynamic_meta = false;

    for b in spec.bindings {
        let expected = layout.get(b.slot as usize).ok_or_else(|| {
            format!(
                "Tensor Mode Mismatch: kernel `{name}` binding slot {} out of range (kernel expects {} bindings)",
                b.slot,
                layout.len()
            )
        })?;
        if !expected.accepts(&b.mode) {
            return Err(format!(
                "Tensor Mode Mismatch: kernel `{name}` slot {} expects {:?}, got {:?}",
                b.slot, expected, b.mode
            ));
        }

        covered[b.slot as usize] = true;
        if let BindingRole::Meta {
            kind: MetaKind::Dynamic,
            ..
        } = b.mode
        {
            has_dynamic_meta = true;
        }
    }

    if !covered.iter().all(|&c| c) {
        return Err(format!(
            "Tensor Mode Mismatch: kernel `{name}` expects {} binding(s), only {} were supplied",
            layout.len(),
            covered.iter().filter(|&&c| c).count()
        ));
    }
    Ok(has_dynamic_meta)
}

fn check_hazards<Buf: Buffer>(specs: &[NodeSpec<Buf>]) -> Result<(), String> {
    let mut last_write: std::collections::HashMap<BufferId, usize> =
        std::collections::HashMap::new();

    for (i, spec) in specs.iter().enumerate() {
        for b in spec.bindings {
            let id = b.buffer.id();

            match b.mode {
                BindingRole::Output(_) | BindingRole::InOut(_) => {
                    if let Some(&prev) = last_write.get(&id) {
                        return Err(format!(
                            "Buffer hazard: dispatch {i} writes a buffer already written by \
                             dispatch {prev} with nothing establishing their order"
                        ));
                    }
                    last_write.insert(id, i);
                }
                BindingRole::Accumulate(_) => {
                    last_write.insert(id, i);
                }
                BindingRole::Input(_) | BindingRole::Meta { .. } => {}
            }
        }
    }
    Ok(())
}

pub enum DispatchPlan {
    Captured {
        key: usize,
        nodes: std::ops::Range<usize>,
    },
    Streamed {
        nodes: std::ops::Range<usize>,
    },
}

pub struct Graph<B: Backend> {
    ctx: std::sync::Arc<B>,
    nodes: Vec<B::Node>,
    plan: Vec<DispatchPlan>,
}

impl<B: Backend> Graph<B> {
    pub fn build(ctx: std::sync::Arc<B>, specs: &[NodeSpec<B::Buffer>]) -> Result<Self, String> {
        for spec in specs {
            validate_spec::<B::Node, _>(spec)?;
        }
        check_hazards(specs)?;

        let nodes: Vec<B::Node> = specs
            .iter()
            .map(|s| ctx.build_node(s.shader, s.bindings, s.workgroups))
            .collect();
        let plan = vec![DispatchPlan::Streamed {
            nodes: 0..nodes.len(),
        }];

        Ok(Self { ctx, nodes, plan })
    }

    pub fn run(&self) {
        for segment in &self.plan {
            match segment {
                DispatchPlan::Streamed { nodes } => self.ctx.execute(&self.nodes[nodes.clone()]),
                DispatchPlan::Captured { key, nodes } => {
                    self.ctx.execute_captured(*key, &self.nodes[nodes.clone()])
                }
            }
        }
    }

    pub fn capture(&mut self, key: usize, range: std::ops::Range<usize>) {
        let mut new_plan = Vec::with_capacity(self.plan.len() + 2);
        for segment in std::mem::take(&mut self.plan) {
            match segment {
                DispatchPlan::Streamed { nodes } => {
                    let overlap_start = nodes.start.max(range.start);
                    let overlap_end = nodes.end.min(range.end);
                    if overlap_start >= overlap_end {
                        new_plan.push(DispatchPlan::Streamed { nodes });
                        continue;
                    }
                    if nodes.start < overlap_start {
                        new_plan.push(DispatchPlan::Streamed {
                            nodes: nodes.start..overlap_start,
                        });
                    }
                    new_plan.push(DispatchPlan::Captured {
                        key,
                        nodes: overlap_start..overlap_end,
                    });
                    if overlap_end < nodes.end {
                        new_plan.push(DispatchPlan::Streamed {
                            nodes: overlap_end..nodes.end,
                        });
                    }
                }
                already_captured => new_plan.push(already_captured),
            }
        }
        self.plan = new_plan;
    }
}

#[cfg(test)]
mod smoke {
    use super::*;
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
        fn id(&self) -> BufferId {
            BufferId(StdArc::as_ptr(&self.0) as u64)
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

    struct ToyBackend;
    impl Storage for ToyBackend {
        type Buffer = ToyBuffer;
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
            ToyBuffer(StdArc::new(Mutex::new(vec![0u8; elem_count * 4])))
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

    impl Device for ToyDevice {
        fn label(&self) -> String {
            format!("toy-device-{}", self.0)
        }
        fn total_memory_bytes(&self) -> u64 {
            1024
        }
    }

    impl Topology for ToyBackend {
        type Device = ToyDevice;
        fn choosable_devices() -> Vec<Self::Device> {
            vec![ToyDevice(0), ToyDevice(1)]
        }
        fn attach(_device: Self::Device) -> Result<Self, String> {
            Ok(ToyBackend)
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
            ToyBuffer(StdArc::new(Mutex::new(vec![0u8; elem_count * 4])))
        }
    }

    #[test]
    fn carve_from_area() {
        let ctx = StdArc::new(ToyBackend);
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

    #[test]
    fn graph_build_and_run() {
        let ctx = StdArc::new(ToyBackend);
        let a = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; 4])));
        let b = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; 4])));
        let bindings = [
            Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
            Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
        ];
        let specs = [NodeSpec {
            shader: &COPY_SHADER,
            bindings: &bindings,
            workgroups: Workgroups::linear(1),
        }];
        let graph = Graph::build(ctx, &specs).unwrap();
        graph.run();
    }

    #[test]
    fn graph_capture_falls_back_to_execute() {
        let ctx = StdArc::new(ToyBackend);
        let a = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; 4])));
        let b = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; 4])));
        let bindings = [
            Binding::new(0, &a, BindingRole::Input(DataKind::F32)),
            Binding::new(1, &b, BindingRole::Output(DataKind::F32)),
        ];
        let specs = [NodeSpec {
            shader: &COPY_SHADER,
            bindings: &bindings,
            workgroups: Workgroups::linear(1),
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
        let ctx = StdArc::new(ToyBackend);
        let a = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; 4])));
        let b = ToyBuffer(StdArc::new(Mutex::new(vec![0u8; 4])));
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
                shader: &COPY_SHADER,
                bindings: &bindings1,
                workgroups: Workgroups::linear(1),
            },
            NodeSpec {
                shader: &COPY_SHADER,
                bindings: &bindings2,
                workgroups: Workgroups::linear(1),
            },
        ];
        let err = Graph::build(ctx, &specs).err().unwrap();
        assert!(err.contains("Buffer hazard"), "unexpected error: {err}");
    }
}
