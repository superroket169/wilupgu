use super::*;
use crate::backend::tests::{copy_node, device_with};

#[test]
fn graph_build_and_run() {
    let (ctx, ids) = device_with(2);
    let graph = Graph::build(ctx, &[copy_node(ids[0], ids[1])]).unwrap();
    graph.run();
}

#[test]
fn graph_capture_falls_back_to_execute() {
    let (ctx, ids) = device_with(2);
    let mut graph = Graph::build(ctx, &[copy_node(ids[0], ids[1])]).unwrap();
    graph.capture(7, 0..1);
    assert!(matches!(
        graph.plan.as_slice(),
        [DispatchPlan::Captured { key: 7, .. }]
    ));
    graph.run(); // ToyBackend has no real capture support, runs via the default fallback
}

#[test]
fn graph_build_rejects_unordered_double_write() {
    let (ctx, ids) = device_with(2);
    let specs = [copy_node(ids[0], ids[1]), copy_node(ids[0], ids[1])];
    let err = Graph::build(ctx, &specs).err().unwrap();
    assert!(err.contains("Buffer hazard"), "unexpected error: {err}");
}

#[test]
fn graph_build_rejects_foreign_buffer() {
    let (ctx, ids) = device_with(1);
    let (_other, foreign) = device_with(1); // allocated on a different device
    let err = Graph::build(ctx, &[copy_node(ids[0], foreign[0])])
        .err()
        .unwrap();
    assert!(
        err.contains("Buffer ownership mismatch"),
        "unexpected error: {err}"
    );
}
