//! A backend for compile time & build time rule tests
//!
//! buffers are host `Vec<u8>`s and `execute` does nothing.
//! It implements every trait in `backend.rs`, so the contract's tests run on it.
//!
//! this file in cfg(test), look mod.rs

use std::sync::{Arc, Mutex};

use crate::backend::{
    Area, Areable, Buffer, DeviceInfo, Dispatch, Node, Storage, SupportsCarve, SupportsDType,
    Topology,
};
use crate::core::device::DeviceId;
use crate::core::dtype::{DataKind, DataType, HostData, F32};
use crate::core::node::Binding;
use crate::core::shader::{Shader, ShaderFormat, Workgroups};
use crate::core::table::BufferTable;

#[derive(Clone)]
pub(crate) struct ToyBuffer(pub(crate) Arc<Mutex<Vec<u8>>>);
impl Buffer for ToyBuffer {
    fn size_bytes(&self) -> u64 {
        self.0.lock().unwrap().len() as u64
    }
    fn holders(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

#[derive(Clone)]
pub(crate) struct ToyNode {
    pub(crate) shader: &'static Shader,
    pub(crate) workgroups: Workgroups,
}
impl Node for ToyNode {
    fn shader(&self) -> &'static Shader {
        self.shader
    }
    fn workgroups(&self) -> Workgroups {
        self.workgroups
    }
}

pub(crate) struct ToyBackend {
    id: DeviceId,
    table: BufferTable<ToyBuffer>,
    // every `update_meta` call's words, in order
    pub(crate) meta_writes: Mutex<Vec<Vec<u32>>>,
}
impl ToyBackend {
    pub(crate) fn new() -> Self {
        Self {
            id: DeviceId::new(),
            table: BufferTable::new(1024),
            meta_writes: Mutex::new(Vec::new()),
        }
    }
}
impl Storage for ToyBackend {
    type Buffer = ToyBuffer;
    fn device_id(&self) -> DeviceId {
        self.id
    }
    fn table(&self) -> &BufferTable<ToyBuffer> {
        &self.table
    }
    fn alloc_kind(&self, kind: DataKind, elem_count: usize) -> Result<ToyBuffer, String> {
        match kind {
            DataKind::F32 => Ok(SupportsDType::<F32>::alloc(self, elem_count)),
            _ => Err(format!("toy does not support {kind:?}")),
        }
    }
    fn upload_kind(&self, buf: &ToyBuffer, data: &HostData) -> Result<(), String> {
        match data {
            HostData::F32(v) => Ok(SupportsDType::<F32>::upload(self, buf, v)),
            _ => Err(format!("toy does not support {:?}", data.kind())),
        }
    }
    fn download_kind(
        &self,
        buf: &ToyBuffer,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<HostData, String> {
        match kind {
            DataKind::F32 => Ok(F32::wrap(SupportsDType::<F32>::download(
                self, buf, elem_count,
            ))),
            _ => Err(format!("toy does not support {kind:?}")),
        }
    }
}
impl Dispatch for ToyBackend {
    type Node = ToyNode;
    const FORMAT: ShaderFormat = ShaderFormat::Native;
    fn build_node(
        &self,
        shader: &'static Shader,
        _meta: &[u32],
        _bindings: &[(Binding, ToyBuffer)],
        workgroups: Workgroups,
    ) -> Self::Node {
        ToyNode { shader, workgroups }
    }
    fn update_meta(&self, _node: &Self::Node, meta: &[u32]) {
        self.meta_writes.lock().unwrap().push(meta.to_vec());
    }
    fn execute(&self, _nodes: &[Self::Node]) {}
    fn synchronize(&self) {}
}

// ToyBackend never overrides execute_captured/release_captured --
// exercises Dispatch's default (runtime-fallback) bodies.

// ToyBackend only ever implements SupportsDType<F32> -- on purpose.
impl SupportsDType<F32> for ToyBackend {
    fn alloc(&self, elem_count: usize) -> ToyBuffer {
        ToyBuffer(Arc::new(Mutex::new(vec![0u8; elem_count * 4])))
    }
    fn upload(&self, buf: &ToyBuffer, data: &[f32]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        buf.0.lock().unwrap()[..bytes.len()].copy_from_slice(bytes);
    }
    fn download(&self, buf: &ToyBuffer, elem_count: usize) -> Vec<f32> {
        let bytes = buf.0.lock().unwrap();
        bytemuck::cast_slice(&bytes[..elem_count * 4]).to_vec()
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
    fn carve(&self, _area: &mut Self::Area, elem_count: usize) -> ToyBuffer {
        SupportsDType::<F32>::alloc(self, elem_count)
    }
}
