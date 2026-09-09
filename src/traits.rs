use crate::backends::BackendDispatch;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The role a single binding slot plays in a kernel's argument list -- NOT a
/// property of a Tensor itself (the same Tensor can be bound as Input in one
/// dispatch and InOut in another).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingRole {
    Input,
    Output,
    InOut,
    Accumulate,
    /// `dynamic`: this call's meta buffer changes value between dispatches of
    /// the *same* captured graph (e.g. decode's advancing position), so a
    /// backend must not bake its contents into a cached/captured dispatch.
    Meta {
        fields: MetaSlot,
        dynamic: bool,
    },
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
}

#[derive(Clone, Copy)]
pub struct F32;
impl DataType for F32 {
    type HostRepr = f32;
    const BITS_PER_ELEM: u32 = 32;
}

#[derive(Clone, Copy)]
pub struct F16;
impl DataType for F16 {
    type HostRepr = half::f16;
    const BITS_PER_ELEM: u32 = 16;
}

#[derive(Clone, Copy)]
pub struct Bf16;
impl DataType for Bf16 {
    type HostRepr = half::bf16;
    const BITS_PER_ELEM: u32 = 16;
}

#[derive(Clone, Copy)]
pub struct Int8;
impl DataType for Int8 {
    type HostRepr = u8;
    const BITS_PER_ELEM: u32 = 8;
}

#[derive(Clone, Copy)]
pub struct Int4;
impl DataType for Int4 {
    type HostRepr = u32;
    const BITS_PER_ELEM: u32 = 4;
    const ELEMS_PER_HOST_WORD: u32 = 8;
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

pub trait Backend: Send + Sync + 'static {
    type Buffer: Buffer;
    type Node: Node;

    fn name(&self) -> &'static str;

    fn free_buffer(&self, buf: Self::Buffer);
    fn recycle(&self, buf: Self::Buffer);

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<Self::Buffer>],
        workgroups: Workgroups,
    ) -> Self::Node;
    fn execute(&self, nodes: &[Self::Node]);
    fn execute_captured(&self, _key: usize, nodes: &[Self::Node]) {
        self.execute(nodes);
    }
    fn release_captured(&self, _key: usize) {}
    fn synchronize(&self);
}

pub trait SupportsDType<D: DataType>: Backend {
    fn alloc(&self, elem_count: usize) -> Self::Buffer;
    fn upload(&self, buf: &Self::Buffer, data: &[D::HostRepr]);
    fn download(&self, buf: &Self::Buffer) -> Vec<D::HostRepr>;
}

pub struct Tensor<B: SupportsDType<D>, D: DataType> {
    pub ctx: Arc<B>,
    pub buffer: B::Buffer,
    pub elem_count: usize,
    _dtype: PhantomData<D>,
}

impl<B: SupportsDType<D>, D: DataType> Tensor<B, D> {
    pub fn new(ctx: Arc<B>, elem_count: usize) -> Self {
        let buffer = ctx.alloc(elem_count);
        Self {
            ctx,
            buffer,
            elem_count,
            _dtype: PhantomData,
        }
    }

    pub fn init_from_cpu(ctx: Arc<B>, data: &[D::HostRepr]) -> Self {
        let elem_count = data.len() * D::ELEMS_PER_HOST_WORD as usize;
        let buffer = ctx.alloc(elem_count);
        ctx.upload(&buffer, data);
        Self {
            ctx,
            buffer,
            elem_count,
            _dtype: PhantomData,
        }
    }

    fn host_words(&self) -> usize {
        self.elem_count.div_ceil(D::ELEMS_PER_HOST_WORD as usize)
    }

    pub fn copy_from_cpu(&self, data: &[D::HostRepr]) {
        assert_eq!(
            data.len(),
            self.host_words(),
            "Tensor::copy_from_cpu: buffer holds {} host word(s) but {} were given",
            self.host_words(),
            data.len()
        );
        self.ctx.upload(&self.buffer, data);
    }

    pub fn to_cpu(&self) -> Vec<D::HostRepr> {
        let mut v = self.ctx.download(&self.buffer);
        v.truncate(self.host_words());
        v
    }

    pub fn perm_drop(self) {
        let this = std::mem::ManuallyDrop::new(self);

        let ctx = unsafe { std::ptr::read(&this.ctx) };
        let buffer = unsafe { std::ptr::read(&this.buffer) };
        ctx.free_buffer(buffer);
    }
}

impl<B: SupportsDType<D>, D: DataType> Drop for Tensor<B, D> {
    fn drop(&mut self) {
        if self.buffer.is_sole_owner() {
            self.ctx.recycle(self.buffer.clone());
        }
    }
}

static NEXT_GRAPH_ID: AtomicUsize = AtomicUsize::new(0);

pub struct ComputeGraph<B: Backend> {
    ctx: Arc<B>,
    id: usize,
    nodes: Vec<B::Node>,
}

impl<B: Backend> ComputeGraph<B> {
    pub fn new(ctx: Arc<B>) -> Self {
        Self {
            ctx,
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::new(),
        }
    }

    pub fn add_node(
        &mut self,
        shader: &'static Shader,
        bindings: &[Binding<B::Buffer>],
        workgroups: Workgroups,
    ) {
        B::Node::validate_workgroups(workgroups)
            .unwrap_or_else(|e| panic!("kernel `{}`: {e}", shader.name));

        let layout = shader.layout;
        let name = shader.name;
        let mut covered = vec![false; layout.len()];
        for b in bindings {
            let expected = layout.get(b.slot as usize).unwrap_or_else(|| {
                panic!(
                    "Tensor Mode Mismatch: kernel `{name}` binding slot {} out of range (kernel expects {} bindings)",
                    b.slot,
                    layout.len()
                )
            });
            assert_eq!(
                std::mem::discriminant(expected),
                std::mem::discriminant(&b.mode),
                "Tensor Mode Mismatch: kernel `{name}` slot {} expects {:?}, got {:?}",
                b.slot,
                expected,
                b.mode
            );
            covered[b.slot as usize] = true;
        }
        assert!(
            covered.iter().all(|&c| c),
            "Tensor Mode Mismatch: kernel `{name}` expects {} binding(s), only {} were supplied",
            layout.len(),
            covered.iter().filter(|&&c| c).count()
        );

        let node = self.ctx.build_node(shader, bindings, workgroups);
        self.nodes.push(node);
    }

    pub fn execute(&self) {
        self.ctx.execute(&self.nodes);
    }

    pub fn execute_captured(&self) {
        self.ctx.execute_captured(self.id, &self.nodes);
    }
}

impl<B: Backend> Drop for ComputeGraph<B> {
    fn drop(&mut self) {
        self.ctx.release_captured(self.id);
    }
}

pub fn fuse_compute_graphs<B: Backend>(
    ctx: Arc<B>,
    graphs: &[&ComputeGraph<B>],
) -> ComputeGraph<B> {
    let nodes = graphs
        .iter()
        .flat_map(|g| g.nodes.iter().cloned())
        .collect();
    ComputeGraph {
        ctx,
        id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
        nodes,
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
    impl Backend for ToyBackend {
        type Buffer = ToyBuffer;
        type Node = ToyNode;
        fn name(&self) -> &'static str {
            "toy"
        }
        fn free_buffer(&self, _buf: Self::Buffer) {}
        fn recycle(&self, _buf: Self::Buffer) {}
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

    #[test]
    fn f32_roundtrip() {
        let ctx = StdArc::new(ToyBackend);
        let t: Tensor<ToyBackend, F32> = Tensor::init_from_cpu(ctx, &[1.0, 2.0, 3.0]);
        assert_eq!(t.to_cpu(), vec![1.0, 2.0, 3.0]);
    }

    // Uncommenting this must NOT compile -- ToyBackend never implements
    // SupportsDType<Int8>, so this should be a compile error, not a panic.
    // #[test]
    // fn int8_is_a_compile_error() {
    //     let ctx = StdArc::new(ToyBackend);
    //     let _t: Tensor<ToyBackend, Int8> = Tensor::new(ctx, 4);
    // }
}
