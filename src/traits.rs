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
}

pub trait Backend: Dispatch {}
impl<T: Dispatch> Backend for T {}

pub trait Capturable: Dispatch {
    fn execute_captured(&self, key: usize, nodes: &[Self::Node]);
    fn release_captured(&self, key: usize);
}

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

    impl Capturable for ToyBackend {
        fn execute_captured(&self, _key: usize, nodes: &[Self::Node]) {
            self.execute(nodes);
        }
        fn release_captured(&self, _key: usize) {}
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

}
