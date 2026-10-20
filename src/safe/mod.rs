//! The safe layer: the API users call. Generic over the backend, it holds
//! every check that can be written once for all backends, and it is the only
//! caller of the [`backend`](crate::backend) contract's `unsafe fn`s.
//!
//! Buffers ([`Tensor`]) and nodes ([`Node`]) are made outside a [`Scope`]; GPU
//! work runs inside one, and the scope waits for the device before it returns.

use crate::backend::dtype::{DataKind, HostData};
use crate::backend::io_log;
use crate::backend::pool::BufferPool;
use crate::backend::shader::{BindingRole, MetaType, Shader, Workgroups};
use crate::backend::{
    Area as _, Areable, Backend, Carved, DeviceInfo, Node as _, Topology, COPY_ALIGN,
};

mod scope;
pub use scope::{Arg, NodeSlot, Recording, Scope, TensorSlot};

/// How a [`ComputeDevice`] is set up.
#[derive(Debug, Clone, Default)]
pub struct DeviceConfig {
    /// The most bytes the pool keeps in freed buffers for reuse; 0 turns the
    /// pool off.
    pub pool_bytes: u64,
}

/// One meta field's value. [`ComputeDevice::node`] and [`Node::update_meta`]
/// check each against the type of the shader's field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetaValue {
    /// For a [`MetaType::Uint`] field.
    Uint(u32),
    /// For a [`MetaType::Float`] field.
    Float(f32),
}

/// An opened device: the raw backend and a pool of freed buffers. Buffers,
/// nodes and areas borrow it, so it outlives them.
pub struct ComputeDevice<B: Backend + Topology> {
    /// Declared before `raw` so the pooled buffers are freed while the device is still open.
    pool: BufferPool<B::Buffer>,
    raw: B,
}

impl<B: Backend + Topology> ComputeDevice<B> {
    /// Wraps a raw device opened with [`Topology::attach`].
    pub fn new(raw: B, config: DeviceConfig) -> Self {
        Self {
            pool: BufferPool::new(config.pool_bytes),
            raw,
        }
    }

    /// The raw device. Its methods that can cause undefined behavior are
    /// `unsafe fn`, so lending it out is harmless.
    pub fn raw(&self) -> &B {
        &self.raw
    }

    /// A buffer of `len` elements of `kind`, from the pool or newly allocated.
    /// A packed kind's length is rounded up to whole host words (a multiple of
    /// 8 for `Int4`). Its contents are unspecified.
    ///
    /// Fails if the device is out of memory, even after the pool is emptied.
    pub fn alloc(&self, kind: DataKind, len: usize) -> Result<Tensor<'_, B>, String> {
        let len = whole_words(kind, len);
        let raw = match self.pool.take(kind, len) {
            Some(buf) => buf,
            None => match self.raw.alloc_raw(buffer_bytes(kind, len)) {
                Ok(buf) => buf,
                Err(_) => {
                    drop(self.pool.drain());
                    self.raw.alloc_raw(buffer_bytes(kind, len))?
                }
            },
        };
        Ok(Tensor {
            mem: TensorMem::Alloc(Some(raw)),
            kind,
            len,
            dev: self,
        })
    }

    /// Builds a node that runs `shader` with `meta`, `workgroups` at a time.
    ///
    /// Fails if the shader has no code in the backend's format, `workgroups`
    /// is over the backend's limit, the device doesn't support a kind in the
    /// layout, or `meta` doesn't match the shader's fields in count and type.
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

// The bytes a tensor's buffer is asked for: its elements' bytes, rounded up
// to `COPY_ALIGN` so the whole of it can always be copied.
fn buffer_bytes(kind: DataKind, len: usize) -> u64 {
    kind.bytes_for(len).next_multiple_of(COPY_ALIGN)
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

/// A tensor: a flat buffer, with no shape. It holds the raw buffer, the kind
/// of its elements and their count, and borrows its device, or the area it
/// was cut out of. Dropping it gives an allocated buffer to the pool.
pub struct Tensor<'d, B: Backend + Topology> {
    mem: TensorMem<'d, B>,
    kind: DataKind,
    len: usize,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology> Tensor<'_, B> {
    /// The kind of its elements.
    pub fn kind(&self) -> DataKind {
        self.kind
    }

    /// How many elements it holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if it holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The raw buffer, to read.
    pub fn raw(&self) -> &B::Buffer {
        match &self.mem {
            TensorMem::Alloc(raw) => raw.as_ref().expect("buffer is only taken in drop"),
            TensorMem::Carved(carved) => carved.buffer(),
        }
    }

    fn raw_mut(&mut self) -> &mut B::Buffer {
        match &mut self.mem {
            TensorMem::Alloc(raw) => raw.as_mut().expect("buffer is only taken in drop"),
            TensorMem::Carved(carved) => carved.buffer_mut(),
        }
    }

    /// Writes `data` to the buffer. Fails unless `data` has the buffer's kind
    /// and length.
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
        // Zero-padded to `COPY_ALIGN`; the padding lands past the last element.
        let padded;
        let bytes = if (bytes.len() as u64).is_multiple_of(COPY_ALIGN) {
            bytes
        } else {
            padded = [bytes, &[0; COPY_ALIGN as usize]].concat();
            &padded[..buffer_bytes(self.kind, self.len) as usize]
        };
        let dev = self.dev;

        // SAFETY: kind and length match, so `bytes` is `buffer_bytes(kind, len)`
        // long, a multiple of `COPY_ALIGN`, and that is what `alloc` asked for
        // (a pooled buffer has the same kind and length).
        unsafe { dev.raw.upload_raw(self.raw_mut(), 0, bytes) };
        Ok(())
    }

    /// Reads the whole buffer back. Blocks until the device work queued before
    /// it is done.
    pub fn download(&self) -> HostData {
        let bytes = self.kind.bytes_for(self.len);
        io_log::log("dtoh", "-", bytes);

        // SAFETY: `alloc` asked for `buffer_bytes(kind, len)` bytes, a multiple
        // of `COPY_ALIGN`.
        let mut data = unsafe {
            self.dev
                .raw
                .download_raw(self.raw(), 0, buffer_bytes(self.kind, self.len))
        };
        data.truncate(bytes as usize);
        HostData::from_bytes(self.kind, data)
    }
}

/// Where a buffer's memory comes from.
enum TensorMem<'d, B: Backend + Topology> {
    /// `Some` until `Drop` moves it to the pool.
    Alloc(Option<B::Buffer>),
    /// Part of a `DeviceArea`; its memory goes back with the area.
    Carved(Carved<'d, B::Buffer>),
}

impl<B: Backend + Topology> Drop for Tensor<'_, B> {
    fn drop(&mut self) {
        if let TensorMem::Alloc(raw) = &mut self.mem {
            if let Some(raw) = raw.take() {
                // A full pool hands the buffer back; dropping it frees it.
                drop(self.dev.pool.recycle(self.kind, self.len, raw));
            }
        }
    }
}

impl<B: Backend + Topology + Areable> ComputeDevice<B> {
    /// Reserves one block of at least `bytes` bytes, to cut buffers out of
    /// with [`DeviceArea::alloc`].
    pub fn reserve(&self, bytes: u64) -> DeviceArea<'_, B> {
        DeviceArea {
            raw: Some(self.raw.reserve(bytes)),
            dev: self,
        }
    }
}

/// One block of device memory, reserved up front. Buffers cut out of it
/// borrow it; dropping it gives the block back to the device.
pub struct DeviceArea<'d, B: Backend + Topology + Areable> {
    // `Some` until `Drop` gives it back to the device.
    raw: Option<B::Area>,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology + Areable> DeviceArea<'_, B> {
    fn raw(&self) -> &B::Area {
        self.raw.as_ref().expect("area is only taken in drop")
    }

    /// The whole area's size, in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.raw().size_bytes()
    }

    /// Bytes not yet cut out.
    pub fn remaining_bytes(&self) -> u64 {
        self.raw().remaining_bytes()
    }

    /// A buffer of `len` elements of `kind`, cut out of the area, with
    /// [`ComputeDevice::alloc`]'s length rule. Its memory goes back with the
    /// area, not to the pool. Fails if the area has no room left.
    pub fn alloc(&self, kind: DataKind, len: usize) -> Result<Tensor<'_, B>, String> {
        let len = whole_words(kind, len);
        let carved = self.dev.raw.carve(self.raw(), buffer_bytes(kind, len))?;
        Ok(Tensor {
            mem: TensorMem::Carved(carved),
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

/// One dispatch of a shader, built by [`ComputeDevice::node`]: the raw node
/// and the elements each slot needs for its current meta.
pub struct Node<'d, B: Backend + Topology> {
    raw: B::Node,
    /// The shader's `min_lens` for the current meta, per slot.
    lens: Vec<usize>,
    dev: &'d ComputeDevice<B>,
}

impl<B: Backend + Topology> Node<'_, B> {
    /// The shader it runs.
    pub fn shader(&self) -> &'static Shader {
        self.raw.shader()
    }

    /// How many workgroups one run dispatches.
    pub fn workgroups(&self) -> Workgroups {
        self.raw.workgroups()
    }

    /// The raw node, to read.
    pub fn raw(&self) -> &B::Node {
        &self.raw
    }

    /// Sets new meta values, checked like [`ComputeDevice::node`]'s. The next
    /// run uses them, and so does the next replay of a recording it is in.
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
