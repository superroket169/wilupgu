use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;

const MAX_FREE_PER_CLASS: usize = 8;

pub(crate) fn size_class(bytes: u64) -> u64 {
    bytes.next_power_of_two()
}

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

    #[must_use]
    pub(crate) fn recycle(&self, key: K, buf: Buf) -> Option<Buf> {
        let mut map = self.free_blocks.lock().unwrap();
        let bucket = map.entry(key).or_default();
        if bucket.len() >= MAX_FREE_PER_CLASS {
            return Some(buf);
        }
        bucket.push(buf);
        None
    }
}
