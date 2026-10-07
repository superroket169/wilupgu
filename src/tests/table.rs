use super::*;
use crate::backend::dtype::HostData;
use crate::backend::{Storage, SupportsDType};
use crate::backends::toy::ToyBackend;
use std::sync::Arc;

#[test]
fn insert_refuses_an_id_already_in_the_table() {
    let toy = ToyBackend::new();
    let table = BufferTable::new(1024);
    let id = TensorId::new();
    table.insert(id, DataKind::F32, 1, toy.alloc(1)).unwrap();
    let err = table
        .insert(id, DataKind::F32, 1, toy.alloc(1))
        .unwrap_err();
    assert!(err.contains("already has a buffer"), "{err}");
}

#[test]
fn remove_sends_an_unheld_buffer_to_the_pool() {
    let toy = ToyBackend::new();
    let table = BufferTable::new(1024);
    let id = TensorId::new();
    let buf = toy.alloc(4);
    table.insert(id, DataKind::F32, 4, buf.clone()).unwrap();
    drop(buf);
    table.remove(id);
    assert!(!table.contains(id));
    assert!(table.take_free(DataKind::F32, 4).is_some());
}

#[test]
fn remove_leaves_a_held_buffer_out_of_the_pool() {
    let toy = ToyBackend::new();
    let table = BufferTable::new(1024);
    let id = TensorId::new();
    let held = toy.alloc(4); // stands in for a built node's clone
    table.insert(id, DataKind::F32, 4, held.clone()).unwrap();
    table.remove(id);
    assert!(table.take_free(DataKind::F32, 4).is_none());
    assert_eq!(held.holders(), 1, "the holder still has its memory");
}

#[test]
fn alloc_reuses_a_freed_buffer() {
    let toy = ToyBackend::new();
    let (a, b) = (TensorId::new(), TensorId::new());
    toy.table().alloc(&toy, a, DataKind::F32, 4).unwrap();
    let first = Arc::as_ptr(&toy.table().get(a).unwrap().0);
    toy.table().remove(a);
    toy.table().alloc(&toy, b, DataKind::F32, 4).unwrap();
    assert_eq!(Arc::as_ptr(&toy.table().get(b).unwrap().0), first);
}

#[test]
fn alloc_refuses_an_id_already_in_the_table() {
    let toy = ToyBackend::new();
    let id = TensorId::new();
    toy.table().alloc(&toy, id, DataKind::F32, 1).unwrap();
    assert!(toy.table().alloc(&toy, id, DataKind::F32, 1).is_err());
}

#[test]
fn upload_and_download_go_through_the_table() {
    let toy = ToyBackend::new();
    let id = TensorId::new();
    toy.table().alloc(&toy, id, DataKind::F32, 2).unwrap();
    toy.table()
        .upload(&toy, id, &HostData::F32(vec![1.0, 2.0]))
        .unwrap();
    let back = toy.table().download(&toy, id, DataKind::F32, 2).unwrap();
    assert!(matches!(back, HostData::F32(v) if v == vec![1.0, 2.0]));
}
