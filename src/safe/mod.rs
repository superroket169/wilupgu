use crate::backend::dtype::{DataKind, HostData};
use crate::backend::io_log;
use crate::backend::pool::BufferPool;
use crate::backend::shader::{BindingRole, MetaType, Shader, Workgroups};
use crate::backend::{Area as _, Areable, Backend, Carved, DeviceInfo, Node as _, Topology};

mod scope;
pub use scope::{Arg, BufSlot, NodeSlot, Recording, Scope};

#[derive(Debug, Clone, Default)]
pub struct DeviceConfig {
    /// 0 turns the pool off.
    pub pool_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetaValue {
    Uint(u32),
    Float(f32),
}

pub struct ComputeDevice<B: Backend + Topology> {
    /// Declared before `raw` so the pooled buffers are freed while the device is still open.
    pool: BufferPool<B::Buffer>,
    raw: B,
}

impl<B: Backend + Topology> ComputeDevice<B> {
    pub fn new(raw: B, config: DeviceConfig) -> Self {
        Self {
            pool: BufferPool::new(config.pool_bytes),
            raw,
        }
    }

    pub fn raw(&self) -> &B {
        &self.raw
    }

    pub fn alloc(&self, kind: DataKind, len: usize) -> Result<Buf<'_, B>, String> {
        let len = whole_words(kind, len);
        let raw = match self.pool.take(kind, len) {
            Some(buf) => buf,
            None => match self.raw.alloc_raw(kind.bytes_for(len)) {
                Ok(buf) => buf,
                Err(_) => {
                    drop(self.pool.drain());
                    self.raw.alloc_raw(kind.bytes_for(len))?
                }
            },
        };
        Ok(Buf {
            mem: BufMem::Alloc(Some(raw)),
            kind,
            len,
            dev: self,
        })
    }

    pub fn node(
        &self,
        shader: &'static Shader,
        meta: &[MetaValue],
        workgroups: Workgroups,
    ) -> Result<Node<'_, B>, String> {
        if !shader.shader_code.has(B::FORMAT) {
            return Err(format!(
                "shader `{}` has no {:?} code",
                shader.name,
                B::FORMAT
            ));
        }
        B::Node::validate_workgroups(workgroups)?;
        for role in shader.layout {
            let (BindingRole::Input(kind)
            | BindingRole::Output(kind)
            | BindingRole::InOut(kind)
            | BindingRole::Accumulate(kind)) = role;
            if !self.raw.info().supports_kind(*kind) {
                return Err(format!(
                    "shader `{}`: device `{}` doesn't support {kind:?}",
                    shader.name,
                    self.raw.info().label()
                ));
            }
        }
        let words = meta_words(shader, meta)?;
        Ok(Node {
            raw: self.raw.build_node(shader, &words, workgroups),
            lens: min_lens(shader, &words, workgroups),
            dev: self,
        })
    }
}

// A packed kind's length is rounded up to whole host words.
fn whole_words(kind: DataKind, len: usize) -> usize {
    len.next_multiple_of(kind.elems_per_host_word())
}

fn min_lens(shader: &Shader, words: &[u32], workgroups: Workgroups) -> Vec<usize> {
    let mut lens = vec![0; shader.layout.len()];
    (shader.min_lens)(words, workgroups, &mut lens);
    lens
}

fn meta_words(shader: &Shader, meta: &[MetaValue]) -> Result<Vec<u32>, String> {
    if meta.len() != shader.meta.len() {
        return Err(format!(
            "shader `{}`: expected {} meta fields, got {}",
            shader.name,
            shader.meta.len(),
            meta.len()
        ));
    }
    shader
        .meta
        .iter()
        .zip(meta)
        .map(|(field, value)| match (field.ty, value) {
            (MetaType::Uint, MetaValue::Uint(v)) => Ok(*v),
            (MetaType::Float, MetaValue::Float(v)) => Ok(v.to_bits()),
            _ => Err(format!(
                "shader `{}`: meta field `{}` is {:?}, got {value:?}",
                shader.name, field.name, field.ty
            )),
        })
        .collect()
}

pub struct Buf<'d, B: Backend + Topology> {
    mem: BufMem<'d, B>,
    kind: DataKind,
    len: usize,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology> Buf<'_, B> {
    pub fn kind(&self) -> DataKind {
        self.kind
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn raw(&self) -> &B::Buffer {
        match &self.mem {
            BufMem::Alloc(raw) => raw.as_ref().expect("buffer is only taken in drop"),
            BufMem::Carved(carved) => carved.buffer(),
        }
    }

    fn raw_mut(&mut self) -> &mut B::Buffer {
        match &mut self.mem {
            BufMem::Alloc(raw) => raw.as_mut().expect("buffer is only taken in drop"),
            BufMem::Carved(carved) => carved.buffer_mut(),
        }
    }

    pub fn upload(&mut self, data: &HostData) -> Result<(), String> {
        if data.kind() != self.kind || data.len() != self.len {
            return Err(format!(
                "upload of {} {:?} elements into a buffer of {} {:?}",
                data.len(),
                data.kind(),
                self.len,
                self.kind
            ));
        }
        let bytes = data.as_bytes();
        io_log::log("htod", "-", bytes.len() as u64);
        let dev = self.dev;

        // SAFETY: kind and length match, so `bytes` is `kind.bytes_for(len)`,
        // which `alloc` asked for (a pooled buffer has the same kind and length).
        unsafe { dev.raw.upload_raw(self.raw_mut(), bytes) };
        Ok(())
    }

    pub fn download(&self) -> HostData {
        let bytes = self.kind.bytes_for(self.len);
        io_log::log("dtoh", "-", bytes);

        // SAFETY: `alloc` asked for `kind.bytes_for(len)` bytes.
        let data = unsafe { self.dev.raw.download_raw(self.raw(), bytes) };
        HostData::from_bytes(self.kind, data)
    }
}

/// Where a buffer's memory comes from.
enum BufMem<'d, B: Backend + Topology> {
    /// `Some` until `Drop` moves it to the pool.
    Alloc(Option<B::Buffer>),
    /// Part of a `DeviceArea`; its memory goes back with the area.
    Carved(Carved<'d, B::Buffer>),
}

impl<B: Backend + Topology> Drop for Buf<'_, B> {
    fn drop(&mut self) {
        if let BufMem::Alloc(raw) = &mut self.mem {
            if let Some(raw) = raw.take() {
                // A full pool hands the buffer back; dropping it frees it.
                drop(self.dev.pool.recycle(self.kind, self.len, raw));
            }
        }
    }
}

impl<B: Backend + Topology + Areable> ComputeDevice<B> {
    pub fn reserve(&self, bytes: u64) -> DeviceArea<'_, B> {
        DeviceArea {
            raw: Some(self.raw.reserve(bytes)),
            dev: self,
        }
    }
}

pub struct DeviceArea<'d, B: Backend + Topology + Areable> {
    // `Some` until `Drop` gives it back to the device.
    raw: Option<B::Area>,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology + Areable> DeviceArea<'_, B> {
    fn raw(&self) -> &B::Area {
        self.raw.as_ref().expect("area is only taken in drop")
    }

    pub fn size_bytes(&self) -> u64 {
        self.raw().size_bytes()
    }

    pub fn remaining_bytes(&self) -> u64 {
        self.raw().remaining_bytes()
    }

    pub fn alloc(&self, kind: DataKind, len: usize) -> Result<Buf<'_, B>, String> {
        let len = whole_words(kind, len);
        let carved = self.dev.raw.carve(self.raw(), kind.bytes_for(len))?;
        Ok(Buf {
            mem: BufMem::Carved(carved),
            kind,
            len,
            dev: self.dev,
        })
    }
}

impl<B: Backend + Topology + Areable> Drop for DeviceArea<'_, B> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            self.dev.raw.release(raw);
        }
    }
}

pub struct Node<'d, B: Backend + Topology> {
    raw: B::Node,
    /// The shader's `min_lens` for the current meta, per slot.
    lens: Vec<usize>,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology> Node<'_, B> {
    pub fn shader(&self) -> &'static Shader {
        self.raw.shader()
    }

    pub fn workgroups(&self) -> Workgroups {
        self.raw.workgroups()
    }

    pub fn raw(&self) -> &B::Node {
        &self.raw
    }

    pub fn update_meta(&mut self, meta: &[MetaValue]) -> Result<(), String> {
        let shader = self.raw.shader();
        let words = meta_words(shader, meta)?;
        self.lens = min_lens(shader, &words, self.raw.workgroups());
        self.dev.raw.update_meta(&mut self.raw, &words);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/safe.rs"]
mod tests;
