use crate::backends::BackendDispatch;
use crate::resolver::Resolvable;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum MetaKind {
    Static,
    Dynamic,
    Maximized(Resolvable<u32>),
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceId(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TensorSpecId(u64);

impl TensorSpecId {
    fn next() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

pub enum TensorSize {
    Fixed(u32),
    /// Not known in compile time
    /// whoever resolves it decides HOW based on the owning NodeSpec's `Parallelity`
    Resolvable {
        size: Resolvable<u32>,
        multiplier: u32,
        coefficient: u32,
    },
}

pub enum InitRecipe<D: DataType> {
    UploadFromHost(Vec<D::HostRepr>),
    Zero,
}

pub struct TensorSpec<D: DataType> {
    id: TensorSpecId,
    size: TensorSize,
    init: Option<InitRecipe<D>>,
}

impl<D: DataType> TensorSpec<D> {
    pub fn blank(size: TensorSize) -> Self {
        Self {
            id: TensorSpecId::next(),
            size,
            init: None,
        }
    }

    pub fn seeded(size: TensorSize, init: InitRecipe<D>) -> Self {
        Self {
            id: TensorSpecId::next(),
            size,
            init: Some(init),
        }
    }

    pub fn id(&self) -> TensorSpecId {
        self.id
    }

    pub fn size(&self) -> &TensorSize {
        &self.size
    }

    pub fn init(&self) -> Option<&InitRecipe<D>> {
        self.init.as_ref()
    }
}

pub trait Buffer: Clone + Send + Sync + 'static {
    fn size_bytes(&self) -> u64;
    fn is_sole_owner(&self) -> bool;
    fn id(&self) -> BufferId;
    fn owner(&self) -> DeviceId;
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
    fn device_id(&self) -> DeviceId;
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

    fn supports_p2p(&self, other: DeviceId) -> bool {
        let _ = other;
        false
    }
}

pub trait Backend: Dispatch {}
impl<T: Dispatch> Backend for T {}

pub trait SupportsDType<D: DataType>: Storage {
    fn alloc(&self, elem_count: usize) -> Self::Buffer;
    fn upload(&self, buf: &Self::Buffer, data: &[D::HostRepr]);
    fn download(&self, buf: &Self::Buffer) -> Vec<D::HostRepr>;

    fn copy_to(&self, buf: &Self::Buffer, dest: &Self) -> Self::Buffer {
        let data = self.download(buf);
        let new_buf = dest.alloc(data.len());
        dest.upload(&new_buf, &data);
        new_buf
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
    fn carve(&self, area: &mut Self::Area, elem_count: usize) -> Self::Buffer;
}

pub struct NodeSpec<'a, Buf> {
    pub shader: &'static Shader,
    pub bindings: &'a [Binding<'a, Buf>],
    pub workgroups: Workgroups,
    pub parallelity: crate::mesh::Parallelity,
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
        } = &b.mode
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

fn check_ownership<B: Backend>(ctx: &B, specs: &[NodeSpec<B::Buffer>]) -> Result<(), String> {
    let owner = ctx.device_id();
    for (i, spec) in specs.iter().enumerate() {
        for b in spec.bindings {
            let actual = b.buffer.owner();
            if actual != owner {
                return Err(format!(
                    "Buffer ownership mismatch: dispatch {i} (kernel `{}`) binding slot {} \
                     belongs to device {actual:?}, but this Graph runs on {owner:?}",
                    spec.shader.name, b.slot
                ));
            }
        }
    }
    Ok(())
}

fn check_hazards<Buf: Buffer>(specs: &[NodeSpec<Buf>]) -> Result<(), String> {
    let mut last_write: std::collections::HashMap<BufferId, usize> =
        std::collections::HashMap::new();

    for (i, spec) in specs.iter().enumerate() {
        for b in spec.bindings {
            let id = b.buffer.id();

            match &b.mode {
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
        check_ownership(ctx.as_ref(), specs)?;
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
#[path = "traits_tests.rs"]
mod tests;
