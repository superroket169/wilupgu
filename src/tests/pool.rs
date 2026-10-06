use super::*;
use crate::backends::toy::ToyBuffer;
use std::sync::Arc;

fn buf(bytes: usize) -> ToyBuffer {
    ToyBuffer(Arc::new(Mutex::new(vec![0u8; bytes])))
}

fn same(a: &ToyBuffer, b: &ToyBuffer) -> bool {
    Arc::ptr_eq(&a.0, &b.0)
}

#[test]
fn take_returns_only_the_same_kind_and_length() {
    let pool = BufferPool::new(1024);
    let a = buf(64);
    assert!(pool.recycle(DataKind::F32, 16, a.clone()).is_none());
    assert!(pool.take(DataKind::F16, 16).is_none());
    assert!(pool.take(DataKind::F32, 32).is_none());
    assert!(same(&pool.take(DataKind::F32, 16).unwrap(), &a));
    assert!(pool.take(DataKind::F32, 16).is_none());
}

#[test]
fn recycle_refuses_past_the_byte_limit() {
    let pool = BufferPool::new(100);
    assert!(pool.recycle(DataKind::F32, 15, buf(60)).is_none());
    let b = buf(60);
    assert!(same(
        &pool.recycle(DataKind::F32, 15, b.clone()).unwrap(),
        &b
    ));
    assert!(pool.recycle(DataKind::F32, 10, buf(40)).is_none());
    assert_eq!(pool.free_bytes(), 100);
}

#[test]
fn take_gives_its_bytes_back_to_the_limit() {
    let pool = BufferPool::new(100);
    assert!(pool.recycle(DataKind::F32, 25, buf(100)).is_none());
    pool.take(DataKind::F32, 25).unwrap();
    assert_eq!(pool.free_bytes(), 0);
    assert!(pool.recycle(DataKind::F32, 25, buf(100)).is_none());
}

#[test]
fn drain_empties_everything() {
    let pool = BufferPool::new(1024);
    assert!(pool.recycle(DataKind::F32, 16, buf(64)).is_none());
    assert!(pool.recycle(DataKind::Int8, 32, buf(32)).is_none());
    assert_eq!(pool.drain().len(), 2);
    assert_eq!(pool.free_bytes(), 0);
    assert!(pool.take(DataKind::F32, 16).is_none());
}
