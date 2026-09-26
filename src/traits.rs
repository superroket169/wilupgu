use crate::backends::ShaderCode;
use crate::id::GlobalId;
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

#[derive(Debug, Clone)]
pub struct Binding {
    pub slot: u32,
    pub tensor: TensorId,
    pub mode: BindingRole,
}

impl Binding {
    #[inline]
    pub fn new(slot: u32, tensor: TensorId, mode: BindingRole) -> Self {
        Self { slot, tensor, mode }
    }
}

pub struct Shader {
    pub name: &'static str,
    pub layout: &'static [BindingRole],
    pub dispatch: &'static [ShaderCode],
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

    fn wrap(data: Vec<Self::HostRepr>) -> HostData;
    /// `None` if `data` holds a different kind.
    fn unwrap(data: HostData) -> Option<Vec<Self::HostRepr>>;
}

#[derive(Clone, Copy)]
pub struct F32;
impl DataType for F32 {
    type HostRepr = f32;
    const BITS_PER_ELEM: u32 = 32;
    const KIND: DataKind = DataKind::F32;

    fn wrap(data: Vec<f32>) -> HostData {
        HostData::F32(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<f32>> {
        match data {
            HostData::F32(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct F16;
impl DataType for F16 {
    type HostRepr = half::f16;
    const BITS_PER_ELEM: u32 = 16;
    const KIND: DataKind = DataKind::F16;

    fn wrap(data: Vec<half::f16>) -> HostData {
        HostData::F16(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<half::f16>> {
        match data {
            HostData::F16(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Bf16;
impl DataType for Bf16 {
    type HostRepr = half::bf16;
    const BITS_PER_ELEM: u32 = 16;
    const KIND: DataKind = DataKind::Bf16;

    fn wrap(data: Vec<half::bf16>) -> HostData {
        HostData::Bf16(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<half::bf16>> {
        match data {
            HostData::Bf16(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Int8;
impl DataType for Int8 {
    type HostRepr = u8;
    const BITS_PER_ELEM: u32 = 8;
    const KIND: DataKind = DataKind::Int8;

    fn wrap(data: Vec<u8>) -> HostData {
        HostData::Int8(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<u8>> {
        match data {
            HostData::Int8(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Int4;
impl DataType for Int4 {
    type HostRepr = u32;
    const BITS_PER_ELEM: u32 = 4;
    const ELEMS_PER_HOST_WORD: u32 = 8;
    const KIND: DataKind = DataKind::Int4;

    fn wrap(data: Vec<u32>) -> HostData {
        HostData::Int4(data)
    }

    fn unwrap(data: HostData) -> Option<Vec<u32>> {
        match data {
            HostData::Int4(v) => Some(v),
            _ => None,
        }
    }
}

/// Host-side values of one tensor
/// one tag for the whole vector
/// so kinds can't mix and upload is a plain byte cast.
#[derive(Debug, Clone)]
pub enum HostData {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    Int8(Vec<u8>),
    /// Packed, 8 values per word.
    Int4(Vec<u32>),
}

impl HostData {
    pub fn kind(&self) -> DataKind {
        match self {
            HostData::F32(_) => DataKind::F32,
            HostData::F16(_) => DataKind::F16,
            HostData::Bf16(_) => DataKind::Bf16,
            HostData::Int8(_) => DataKind::Int8,
            HostData::Int4(_) => DataKind::Int4,
        }
    }
}

/// Stand-in tag until the mesh-level `Device` enum exists
/// then `DeviceId` becomes `GlobalId<Device>`.
pub enum DeviceTag {}

pub type DeviceId = GlobalId<DeviceTag>;
pub type TensorId = GlobalId<TensorSpec>;
pub type NodeId = GlobalId<NodeSpec>;

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

pub enum InitRecipe {
    UploadFromHost(HostData),
    Zero,
}

pub struct TensorSpec {
    id: TensorId,
    kind: DataKind,
    size: TensorSize,
    init: Option<InitRecipe>,
}

impl TensorSpec {
    pub fn blank<D: DataType>(size: TensorSize) -> Self {
        Self::with::<D>(size, None)
    }

    pub fn seeded<D: DataType>(size: TensorSize, data: Vec<D::HostRepr>) -> Self {
        Self::with::<D>(size, Some(InitRecipe::UploadFromHost(D::wrap(data))))
    }

    pub fn zeroed<D: DataType>(size: TensorSize) -> Self {
        Self::with::<D>(size, Some(InitRecipe::Zero))
    }

    fn with<D: DataType>(size: TensorSize, init: Option<InitRecipe>) -> Self {
        Self {
            id: GlobalId::new(),
            kind: D::KIND,
            size,
            init,
        }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn kind(&self) -> DataKind {
        self.kind
    }

    pub fn size(&self) -> &TensorSize {
        &self.size
    }

    pub fn init(&self) -> Option<&InitRecipe> {
        self.init.as_ref()
    }
}

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

pub struct NodeSpec {
    id: NodeId,
    pub shader: &'static Shader,
    pub bindings: Vec<Binding>,
    pub workgroups: Workgroups,
    pub parallelity: crate::mesh::Parallelity,
}

impl NodeSpec {
    /// The only way to get a node id
    /// `id` is private so two specs can't be handed the same one.
    pub fn new(
        shader: &'static Shader,
        bindings: Vec<Binding>,
        workgroups: Workgroups,
        parallelity: crate::mesh::Parallelity,
    ) -> Self {
        Self {
            id: GlobalId::new(),
            shader,
            bindings,
            workgroups,
            parallelity,
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }
}

fn validate_spec<N: Node>(spec: &NodeSpec) -> Result<bool, String> {
    N::validate_workgroups(spec.workgroups)
        .map_err(|e| format!("kernel `{}`: {e}", spec.shader.name))?;

    let layout = spec.shader.layout;
    let name = spec.shader.name;
    let mut covered = vec![false; layout.len()];
    let mut has_dynamic_meta = false;

    for b in &spec.bindings {
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

fn check_ownership<B: Backend>(ctx: &B, specs: &[NodeSpec]) -> Result<(), String> {
    for (i, spec) in specs.iter().enumerate() {
        for b in &spec.bindings {
            if !ctx.contains(b.tensor) {
                return Err(format!(
                    "Buffer ownership mismatch: dispatch {i} (kernel `{}`) binding slot {} \
                     names tensor {:?}, which isn't allocated on device {:?}",
                    spec.shader.name,
                    b.slot,
                    b.tensor,
                    ctx.device_id()
                ));
            }
        }
    }
    Ok(())
}

fn check_hazards(specs: &[NodeSpec]) -> Result<(), String> {
    let mut last_write: std::collections::HashMap<TensorId, usize> =
        std::collections::HashMap::new();

    for (i, spec) in specs.iter().enumerate() {
        for b in &spec.bindings {
            let id = b.tensor;

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
    pub fn build(ctx: std::sync::Arc<B>, specs: &[NodeSpec]) -> Result<Self, String> {
        for spec in specs {
            validate_spec::<B::Node>(spec)?;
        }
        check_ownership(ctx.as_ref(), specs)?;
        check_hazards(specs)?;

        let nodes: Vec<B::Node> = specs
            .iter()
            .map(|s| ctx.build_node(s.shader, &s.bindings, s.workgroups))
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
