use std::collections::HashMap;
use std::sync::Mutex;

pub(crate) struct BufferPool<Buf> {
    free_blocks: Mutex<HashMap<u64, Vec<Buf>>>,
}

impl<Buf> BufferPool<Buf> {
    pub(crate) fn new() -> Self {
        Self {
            free_blocks: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn take(&self, size_bytes: u64) -> Option<Buf> {
        self.free_blocks
            .lock()
            .unwrap()
            .get_mut(&size_bytes)
            .and_then(|bucket| bucket.pop())
    }

    pub(crate) fn recycle(&self, size_bytes: u64, buf: Buf) {
        self.free_blocks
            .lock()
            .unwrap()
            .entry(size_bytes)
            .or_default()
            .push(buf);
    }
}
