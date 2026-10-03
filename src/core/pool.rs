use std::collections::HashMap;
use std::sync::Mutex;

use crate::core::dtype::DataKind;

// A free buffer is only handed back for the exact size and kind it was made for.
type Key = (u64, DataKind);

pub(crate) struct BufferPool<Buf> {
    max_free_bytes: u64,
    free: Mutex<Free<Buf>>,
}

struct Free<Buf> {
    blocks: HashMap<Key, Vec<Buf>>,
    bytes: u64,
}

impl<Buf> BufferPool<Buf> {
    pub(crate) fn new(max_free_bytes: u64) -> Self {
        Self {
            max_free_bytes,
            free: Mutex::new(Free {
                blocks: HashMap::new(),
                bytes: 0,
            }),
        }
    }

    // The buffer's old contents are left as they were.
    pub(crate) fn take(&self, size_bytes: u64, kind: DataKind) -> Option<Buf> {
        let mut free = self.free.lock().unwrap();
        let buf = free.blocks.get_mut(&(size_bytes, kind))?.pop()?;
        free.bytes -= size_bytes;
        Some(buf)
    }

    // Hands `buf` back when keeping it would go over `max_free_bytes`;
    // the caller frees it.
    #[must_use]
    pub(crate) fn recycle(&self, size_bytes: u64, kind: DataKind, buf: Buf) -> Option<Buf> {
        let mut free = self.free.lock().unwrap();
        if free.bytes + size_bytes > self.max_free_bytes {
            return Some(buf);
        }
        free.bytes += size_bytes;
        free.blocks.entry((size_bytes, kind)).or_default().push(buf);
        None
    }

    // Empties the pool, for a retry after a failed allocation; the caller frees them.
    #[must_use]
    pub(crate) fn drain(&self) -> Vec<Buf> {
        let mut free = self.free.lock().unwrap();
        free.bytes = 0;
        free.blocks.drain().flat_map(|(_, bufs)| bufs).collect()
    }

    pub(crate) fn free_bytes(&self) -> u64 {
        self.free.lock().unwrap().bytes
    }
}

#[cfg(test)]
#[path = "../tests/pool.rs"]
mod tests;
