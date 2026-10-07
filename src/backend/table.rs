use std::collections::HashMap;
use std::sync::Mutex;

use crate::backend::dtype::{DataKind, HostData};
use crate::backend::id::TensorId;
use crate::backend::pool::BufferPool;
use crate::backend::{Buffer, Storage};

struct Entry<Buf> {
    buf: Buf,
    kind: DataKind,
    elem_count: usize,
}

/// One device's tensors: id -> buffer
pub struct BufferTable<Buf: Buffer> {
    entries: Mutex<HashMap<TensorId, Entry<Buf>>>,
    pool: BufferPool<Buf>,
}

impl<Buf: Buffer> BufferTable<Buf> {
    /// `max_free_bytes` caps how much freed memory the pool keeps for reuse.
    pub fn new(max_free_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            pool: BufferPool::new(max_free_bytes),
        }
    }

    /// Fails instead of replacing: two live tensors must never share one id.
    pub(crate) fn insert(
        &self,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
        buf: Buf,
    ) -> Result<(), String> {
        let mut entries = self.entries.lock().unwrap();
        if entries.contains_key(&id) {
            return Err(format!("tensor {id:?} already has a buffer on this device"));
        }
        entries.insert(
            id,
            Entry {
                buf,
                kind,
                elem_count,
            },
        );
        Ok(())
    }

    pub(crate) fn contains(&self, id: TensorId) -> bool {
        self.entries.lock().unwrap().contains_key(&id)
    }

    /// A clone of `id`'s buffer.
    pub(crate) fn get(&self, id: TensorId) -> Option<Buf> {
        self.entries.lock().unwrap().get(&id).map(|e| e.buf.clone())
    }

    /// Removes `id`. Its buffer goes back to the pool if nobody else holds it;
    /// otherwise it is freed when its last holder (a built node) drops it.
    pub(crate) fn remove(&self, id: TensorId) {
        let Some(entry) = self.entries.lock().unwrap().remove(&id) else {
            return;
        };
        if entry.buf.holders() == 1 {
            // Over the pool's limit, it comes back and is dropped here.
            let _ = self.pool.recycle(entry.kind, entry.elem_count, entry.buf);
        }
    }

    pub(crate) fn take_free(&self, kind: DataKind, elem_count: usize) -> Option<Buf> {
        self.pool.take(kind, elem_count)
    }

    /// A buffer for `id`: from the pool if one fits, else a new one from `storage`.
    pub(crate) fn alloc<S: Storage<Buffer = Buf>>(
        &self,
        storage: &S,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<(), String> {
        if self.contains(id) {
            return Err(format!("tensor {id:?} already has a buffer on this device"));
        }
        let buf = match self.pool.take(kind, elem_count) {
            Some(buf) => buf,
            None => match storage.alloc_kind(kind, elem_count) {
                Ok(buf) => buf,
                // Maybe out of memory: give the pool's buffers back and try once more.
                Err(_) => {
                    drop(self.pool.drain());
                    storage.alloc_kind(kind, elem_count)?
                }
            },
        };
        self.insert(id, kind, elem_count, buf)
    }

    pub(crate) fn upload<S: Storage<Buffer = Buf>>(
        &self,
        storage: &S,
        id: TensorId,
        data: &HostData,
    ) -> Result<(), String> {
        let buf = self.get(id).ok_or_else(|| missing(id))?;
        storage.upload_kind(&buf, data)
    }

    pub(crate) fn download<S: Storage<Buffer = Buf>>(
        &self,
        storage: &S,
        id: TensorId,
        kind: DataKind,
        elem_count: usize,
    ) -> Result<HostData, String> {
        let buf = self.get(id).ok_or_else(|| missing(id))?;
        storage.download_kind(&buf, kind, elem_count)
    }
}

fn missing(id: TensorId) -> String {
    format!("tensor {id:?} has no buffer on this device")
}

#[cfg(test)]
#[path = "../tests/table.rs"]
mod tests;
