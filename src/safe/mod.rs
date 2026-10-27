//! The safe layer: the API users call. Generic over the backend, it holds
//! every check that can be written once for all backends, and it is the only
//! caller of the [`backend`](crate::backend) contract's `unsafe fn`s.
//!
//! Buffers ([`Tensor`]) and nodes ([`Node`]) are made outside a [`Scope`]; GPU
//! work runs inside one, and the scope waits for the device before it returns.

use crate::backend::dtype::{DataKind, DataType, HostData};
use crate::backend::io_log;
use crate::backend::pool::BufferPool;
use crate::backend::shader::{BindingRole, MetaType, Shader, Signature, Workgroups};
use crate::backend::{Area as _, Areable, Backend, Carved, Node as _, Topology, COPY_ALIGN};
use crate::error::Error;

pub(crate) mod rules;
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

impl MetaValue {
    /// The type of field it fills.
    pub fn ty(&self) -> MetaType {
        match self {
            MetaValue::Uint(_) => MetaType::Uint,
            MetaValue::Float(_) => MetaType::Float,
        }
    }
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
    /// Fails if the device doesn't support `kind`, or is out of memory, even
    /// after the pool is emptied.
    pub fn alloc(&self, kind: DataKind, len: usize) -> Result<Tensor<'_, B>, Error> {
        rules::supported_kind(self, kind)?;
        let len = whole_words(kind, len);
        let raw = match self.pool.take(kind, len) {
            Some(buf) => buf,
            None => match self.raw.alloc_raw(buffer_bytes(kind, len)) {
                Ok(buf) => buf,
                Err(_) => {
                    drop(self.pool.drain());
                    self.raw
                        .alloc_raw(buffer_bytes(kind, len))
                        .map_err(Error::Backend)?
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

    /// A tensor holding `data`: [`ComputeDevice::alloc`] then [`Tensor::upload`].
    pub fn tensor_from(&self, data: &HostData) -> Result<Tensor<'_, B>, Error> {
        let mut tensor = self.alloc(data.kind(), data.len())?;
        tensor.upload(data)?;
        Ok(tensor)
    }

    /// Builds a node that runs `shader` with `meta`, `workgroups` at a time.
    ///
    /// Fails if the backend doesn't take the shader's binary or can't load
    /// it, `workgroups` is 0 in some dimension or over the backend's limit,
    /// the device doesn't support a kind in the layout, or `meta` doesn't
    /// match the shader's fields in count and type.
    pub fn node(
        &self,
        shader: &Shader,
        meta: &[MetaValue],
        workgroups: Workgroups,
    ) -> Result<Node<'_, B>, Error> {
        rules::accepted::<B>(shader)?;
        let signature = shader.signature();
        rules::has_workgroups(signature, workgroups)?;

        B::Node::validate_workgroups(workgroups).map_err(Error::Backend)?;

        for role in signature.layout {
            rules::supported_kind(self, role.kind())?;
        }

        let words = meta_words(signature, meta)?;
        Ok(Node {
            raw: self
                .raw
                .build_node(shader, &words, workgroups)
                .map_err(Error::Backend)?,
            lens: min_lens(signature, &words, workgroups),
            meta: meta.to_vec(),
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

fn min_lens(shader: &Signature, words: &[u32], workgroups: Workgroups) -> Vec<usize> {
    let mut lens = vec![0; shader.layout.len()];
    (shader.min_lens)(words, workgroups, &mut lens);
    lens
}

fn meta_words(shader: &Signature, meta: &[MetaValue]) -> Result<Vec<u32>, Error> {
    rules::meta_types(shader, meta.iter().map(MetaValue::ty))?;
    Ok(meta
        .iter()
        .map(|value| match value {
            MetaValue::Uint(v) => *v,
            MetaValue::Float(v) => v.to_bits(),
        })
        .collect())
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

impl<'d, B: Backend + Topology> Tensor<'d, B> {
    /// The device it lives on.
    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.dev
    }

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

    /// Writes `data` to the whole tensor. Fails unless `data` has its kind and length.
    pub fn upload(&mut self, data: &HostData) -> Result<(), Error> {
        rules::same_len(self.len, data.len())?;
        self.upload_at(0, data)
    }

    /// Reads the whole tensor back. Blocks until the device work queued before
    /// it is done.
    pub fn download(&self) -> HostData {
        self.download_range(0, self.len)
            .expect("the whole tensor is always a valid range")
    }

    /// Puts `new` in this tensor's place and gives the old one back. Fails,
    /// changing nothing and giving `new` back, unless `new` is of the same
    /// device and kind. Its length may differ.
    // `new` comes back by value on a fail; replacing is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn replace(&mut self, new: Self) -> Result<Self, (Self, Error)> {
        let fits = rules::same_device(self.dev, new.dev, "tensor")
            .and_then(|()| rules::same_kind(self.kind, new.kind));
        match fits {
            Ok(()) => Ok(std::mem::replace(self, new)),
            Err(e) => Err((new, e)),
        }
    }
}

// Helpers on top of upload and download: shortcuts, typed access, ranges and
// copies. Every range is in elements.
impl<'d, B: Backend + Topology> Tensor<'d, B> {
    /// Writes `data` into the tensor, starting at element `offset`.
    ///
    /// Fails if `data` has another kind, the range doesn't fit, `offset`
    /// doesn't start at a multiple of [`COPY_ALIGN`] bytes, or the range ends
    /// neither at such a multiple nor at the tensor's end.
    pub fn upload_at(&mut self, offset: usize, data: &HostData) -> Result<(), Error> {
        rules::same_kind(self.kind, data.kind())?;
        let range = byte_range(self.kind, self.len, offset, data.len(), true)?;
        let bytes = data.as_bytes();
        io_log::log("htod", bytes.len() as u64);

        // Zero-padded to `COPY_ALIGN`; `byte_range` let it only into the
        // tensor's padding.
        let mut padded = bytes.to_vec();
        padded.resize(range.bytes as usize, 0);
        let dev = self.dev;

        // SAFETY: `byte_range` checked that the range is aligned and inside the
        // `buffer_bytes(kind, len)` that `alloc` asked for (a pooled buffer has
        // the same kind and length).
        unsafe { dev.raw.upload_raw(self.raw_mut(), range.start, &padded) };
        Ok(())
    }

    /// Reads `len` elements back, starting at element `offset`. Blocks until
    /// the device work queued before it is done.
    ///
    /// Fails if the range doesn't fit, or `offset` doesn't start at a multiple
    /// of [`COPY_ALIGN`] bytes.
    pub fn download_range(&self, offset: usize, len: usize) -> Result<HostData, Error> {
        let range = byte_range(self.kind, self.len, offset, len, false)?;
        io_log::log("dtoh", range.exact);

        // SAFETY: `byte_range` checked that the range is aligned and inside the
        // `buffer_bytes(kind, len)` that `alloc` asked for.
        let mut data = unsafe {
            self.dev
                .raw
                .download_raw(self.raw(), range.start, range.bytes)
        };
        data.truncate(range.exact as usize);
        Ok(HostData::from_bytes(self.kind, data))
    }

    /// [`Tensor::upload`] from host words of `D`. Fails unless `D` is the
    /// tensor's kind and `data` its length.
    pub fn upload_slice<D: DataType>(&mut self, data: &[D::HostRepr]) -> Result<(), Error> {
        self.upload(&D::wrap(data.to_vec()))
    }

    /// [`Tensor::download`] as host words of `D`. Fails unless `D` is the
    /// tensor's kind.
    pub fn download_vec<D: DataType>(&self) -> Result<Vec<D::HostRepr>, Error> {
        rules::same_kind(self.kind, D::KIND)?;
        Ok(D::unwrap(self.download()).expect("the download has the tensor's kind"))
    }

    /// A full copy on `dev`, which may be another device or backend. Goes
    /// through the host.
    pub fn copy_to<'e, B2: Backend + Topology>(
        &self,
        dev: &'e ComputeDevice<B2>,
    ) -> Result<Tensor<'e, B2>, Error> {
        dev.tensor_from(&self.download())
    }

    /// Copies `len` elements, starting at element `src_offset` of this tensor,
    /// into `dst` at element `dst_offset`, on the device. Blocks until done.
    ///
    /// Fails if `dst` is on another device (see [`Tensor::copy_to`]) or holds
    /// another kind, or a range breaks [`Tensor::upload_at`]'s rules.
    pub fn copy_into(
        &self,
        src_offset: usize,
        dst: &mut Tensor<'_, B>,
        dst_offset: usize,
        len: usize,
    ) -> Result<(), Error> {
        rules::same_device(self.dev, dst.dev, "tensor")?;
        let (from, to) = copy_ranges(self, src_offset, dst, dst_offset, len)?;

        // SAFETY: `copy_ranges` checked both ranges against `byte_range`'s
        // rules; both tensors are borrowed until `synchronize` returns.
        unsafe {
            self.dev
                .raw
                .copy_raw(self.raw(), from.start, dst.raw_mut(), to.start, to.bytes)
        }
        .map_err(Error::Backend)?;
        self.dev.raw.synchronize();
        Ok(())
    }

    /// Fills the tensor with zero bytes, on the device. Blocks until done.
    pub fn zero(&mut self) -> Result<(), Error> {
        let dev = self.dev;
        // SAFETY: the tensor is borrowed until `synchronize` returns.
        unsafe { dev.raw.clear_raw(self.raw_mut()) }.map_err(Error::Backend)?;
        dev.raw.synchronize();
        Ok(())
    }
}

// A range of elements as `Storage` takes it.
struct ByteRange {
    start: u64,
    // `COPY_ALIGN`-rounded, what is moved.
    bytes: u64,
    // The range's own bytes, what the caller wants.
    exact: u64,
}

// The bytes of elements `offset..offset + len` of a tensor of `tensor_len`.
// The start must be a multiple of `COPY_ALIGN`; the length is rounded up to
// one. A write may only round past its last element at the tensor's end,
// where the rounding falls into the buffer's padding (`buffer_bytes`).
fn byte_range(
    kind: DataKind,
    tensor_len: usize,
    offset: usize,
    len: usize,
    write: bool,
) -> Result<ByteRange, Error> {
    let end = offset
        .checked_add(len)
        .filter(|&end| end <= tensor_len)
        .ok_or(Error::OutOfRange {
            offset,
            len,
            tensor_len,
        })?;
    let bits = kind.bits_per_elem() as u64;
    let align_bits = COPY_ALIGN * 8;
    if !(offset as u64 * bits).is_multiple_of(align_bits) {
        return Err(Error::Misaligned {
            kind,
            element: offset,
        });
    }
    if write && end != tensor_len && !(end as u64 * bits).is_multiple_of(align_bits) {
        return Err(Error::Misaligned { kind, element: end });
    }
    let exact = kind.bytes_for(len);
    Ok(ByteRange {
        start: offset as u64 * bits / 8,
        bytes: exact.next_multiple_of(COPY_ALIGN),
        exact,
    })
}

// The checks of a copy on one device: same kind, `byte_range`'s rules for
// reading `src` and writing `dst`.
fn copy_ranges<B: Backend + Topology>(
    src: &Tensor<'_, B>,
    src_offset: usize,
    dst: &Tensor<'_, B>,
    dst_offset: usize,
    len: usize,
) -> Result<(ByteRange, ByteRange), Error> {
    rules::same_kind(dst.kind, src.kind)?;
    Ok((
        byte_range(src.kind, src.len, src_offset, len, false)?,
        byte_range(dst.kind, dst.len, dst_offset, len, true)?,
    ))
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
    /// area, not to the pool. Fails if the device doesn't support `kind`, or
    /// the area has no room left.
    pub fn alloc(&self, kind: DataKind, len: usize) -> Result<Tensor<'_, B>, Error> {
        rules::supported_kind(self.dev, kind)?;
        let len = whole_words(kind, len);
        let carved = self
            .dev
            .raw
            .carve(self.raw(), buffer_bytes(kind, len))
            .map_err(Error::Backend)?;
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
    meta: Vec<MetaValue>,
    dev: &'d ComputeDevice<B>,
}

impl<'d, B: Backend + Topology> Node<'d, B> {
    /// The device it was built on.
    pub fn device(&self) -> &'d ComputeDevice<B> {
        self.dev
    }

    /// How its shader is called.
    pub fn signature(&self) -> &'static Signature {
        self.raw.signature()
    }

    /// Its meta values, as built or last updated.
    pub fn meta(&self) -> &[MetaValue] {
        &self.meta
    }

    /// How many workgroups one run dispatches.
    pub fn workgroups(&self) -> Workgroups {
        self.raw.workgroups()
    }

    /// The elements each slot needs for the current meta: the shader's
    /// `min_lens`.
    pub fn min_lens(&self) -> &[usize] {
        &self.lens
    }

    /// The raw node, to read.
    pub fn raw(&self) -> &B::Node {
        &self.raw
    }

    /// Sets new meta values, checked like [`ComputeDevice::node`]'s. The next
    /// run uses them, and so does the next replay of a recording it is in.
    pub fn update_meta(&mut self, meta: &[MetaValue]) -> Result<(), Error> {
        let shader = self.raw.signature();
        let words = meta_words(shader, meta)?;
        self.lens = min_lens(shader, &words, self.raw.workgroups());
        self.dev.raw.update_meta(&mut self.raw, &words);
        self.meta = meta.to_vec();
        Ok(())
    }

    /// Puts `new` in this node's place and gives the old one back. Fails,
    /// changing nothing and giving `new` back, unless `new` is of the same
    /// device, its slots hold the same kinds and its meta fields have the
    /// same types. Its shader may differ.
    // `new` comes back by value on a fail; replacing is rare, so its size costs nothing.
    #[allow(clippy::result_large_err)]
    pub fn replace(&mut self, new: Self) -> Result<Self, (Self, Error)> {
        let (old, other) = (self.signature(), new.signature());
        let fits = rules::same_device(self.dev, new.dev, "node")
            .and_then(|()| rules::slot_kinds(other, old.layout.iter().map(BindingRole::kind)))
            .and_then(|()| rules::meta_types(other, old.meta.iter().map(|f| f.ty)));
        match fits {
            Ok(()) => Ok(std::mem::replace(self, new)),
            Err(e) => Err((new, e)),
        }
    }
}

#[cfg(test)]
#[path = "../tests/safe.rs"]
mod tests;
