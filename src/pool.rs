use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;

pub(crate) struct BufferPool<Buf, K: Eq + Hash + Copy = u64> {
    free_blocks: Mutex<HashMap<K, Vec<Buf>>>,
}

impl<Buf, K: Eq + Hash + Copy> BufferPool<Buf, K> {
    pub(crate) fn new() -> Self {
        Self {
            free_blocks: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn take(&self, key: K) -> Option<Buf> {
        self.free_blocks
            .lock()
            .unwrap()
            .get_mut(&key)
            .and_then(|bucket| bucket.pop())
    }

    pub(crate) fn recycle(&self, key: K, buf: Buf) {
        self.free_blocks
            .lock()
            .unwrap()
            .entry(key)
            .or_default()
            .push(buf);
    }
}
