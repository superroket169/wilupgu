use super::*;

#[test]
fn take_returns_only_the_same_size_and_kind() {
    let pool = BufferPool::new(1024);
    assert!(pool.recycle(64, DataKind::F32, "a").is_none());
    assert_eq!(pool.take(64, DataKind::F16), None);
    assert_eq!(pool.take(128, DataKind::F32), None);
    assert_eq!(pool.take(64, DataKind::F32), Some("a"));
    assert_eq!(pool.take(64, DataKind::F32), None);
}

#[test]
fn recycle_refuses_past_the_byte_limit() {
    let pool = BufferPool::new(100);
    assert!(pool.recycle(60, DataKind::F32, "a").is_none());
    assert_eq!(pool.recycle(60, DataKind::F32, "b"), Some("b"));
    assert!(pool.recycle(40, DataKind::F32, "c").is_none());
    assert_eq!(pool.free_bytes(), 100);
}

#[test]
fn take_gives_its_bytes_back_to_the_limit() {
    let pool = BufferPool::new(100);
    assert!(pool.recycle(100, DataKind::F32, "a").is_none());
    pool.take(100, DataKind::F32).unwrap();
    assert_eq!(pool.free_bytes(), 0);
    assert!(pool.recycle(100, DataKind::F32, "b").is_none());
}

#[test]
fn drain_empties_everything() {
    let pool = BufferPool::new(1024);
    assert!(pool.recycle(64, DataKind::F32, "a").is_none());
    assert!(pool.recycle(32, DataKind::Int8, "b").is_none());
    let mut drained = pool.drain();
    drained.sort();
    assert_eq!(drained, vec!["a", "b"]);
    assert_eq!(pool.free_bytes(), 0);
    assert_eq!(pool.take(64, DataKind::F32), None);
}
