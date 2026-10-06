use super::*;
use crate::backend::tests::{copy_node, device_with, META_SHADER, TOY_SHADER};
use crate::backend::Storage;
use crate::core::deferred::{Dynamic, Resolvable};
use crate::core::node::{MetaSource, MetaValue};
use crate::core::shader::Workgroups;

#[test]
fn graph_build_and_run() {
    let (ctx, ids) = device_with(2);
    let graph = Graph::build(&ctx, &[copy_node(ids[0], ids[1])]).unwrap();
    graph.run();
}

#[test]
fn graph_capture_falls_back_to_execute() {
    let (ctx, ids) = device_with(2);
    let mut graph = Graph::build(&ctx, &[copy_node(ids[0], ids[1])]).unwrap();
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
    let err = Graph::build(&ctx, &specs).err().unwrap();
    assert!(err.contains("Buffer hazard"), "unexpected error: {err}");
}

#[test]
fn graph_build_rejects_foreign_buffer() {
    let (ctx, ids) = device_with(1);
    let (_other, foreign) = device_with(1); // allocated on a different device
    let err = Graph::build(&ctx, &[copy_node(ids[0], foreign[0])])
        .err()
        .unwrap();
    assert!(
        err.contains("Buffer ownership mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn graph_build_rejects_missing_shader_code() {
    let (ctx, _) = device_with(0);
    let spec = NodeSpec::new(&TOY_SHADER, vec![], vec![], Workgroups::linear(1));
    let err = Graph::build(&ctx, &[spec]).err().unwrap();
    assert!(
        err.contains("has no Native code"),
        "unexpected error: {err}"
    );
}

fn meta_node(n: MetaSource<u32>, lr: MetaSource<f32>) -> NodeSpec {
    NodeSpec::new(
        &META_SHADER,
        vec![MetaValue::Uint(n), MetaValue::Float(lr)],
        vec![],
        Workgroups::linear(1),
    )
}

#[test]
fn once_only_meta_is_never_rewritten() {
    let (ctx, _) = device_with(0);
    let spec = meta_node(
        MetaSource::Once(Resolvable::fixed(4)),
        MetaSource::Once(Resolvable::fixed(0.5)),
    );
    let graph = Graph::build(&ctx, &[spec]).unwrap();
    graph.run();
    graph.run();
    assert!(ctx.meta_writes.lock().unwrap().is_empty());
}

#[test]
fn per_run_meta_is_written_before_every_run() {
    let (ctx, _) = device_with(0);
    let lr = Dynamic::new();
    let spec = meta_node(
        MetaSource::Once(Resolvable::fixed(4)),
        MetaSource::PerRun(lr.clone()),
    );
    let graph = Graph::build(&ctx, &[spec]).unwrap();
    lr.set(0.5);
    graph.run();
    lr.set(0.25);
    graph.run();
    assert_eq!(
        *ctx.meta_writes.lock().unwrap(),
        vec![vec![4, 0.5f32.to_bits()], vec![4, 0.25f32.to_bits()]]
    );
}

#[test]
fn a_dynamic_shared_by_two_nodes_is_read_once_per_run() {
    let (ctx, _) = device_with(0);
    let lr = Dynamic::new();
    let specs = [
        meta_node(
            MetaSource::Once(Resolvable::fixed(1)),
            MetaSource::PerRun(lr.clone()),
        ),
        meta_node(
            MetaSource::Once(Resolvable::fixed(2)),
            MetaSource::PerRun(lr.clone()),
        ),
    ];
    let graph = Graph::build(&ctx, &specs).unwrap();
    lr.set(0.5);
    graph.run();
    assert_eq!(ctx.meta_writes.lock().unwrap().len(), 2);
}

#[test]
#[should_panic(expected = "without a set since the last run")]
fn running_again_without_setting_a_dynamic_panics() {
    let (ctx, _) = device_with(0);
    let lr = Dynamic::new();
    let spec = meta_node(
        MetaSource::Once(Resolvable::fixed(4)),
        MetaSource::PerRun(lr.clone()),
    );
    let graph = Graph::build(&ctx, &[spec]).unwrap();
    lr.set(0.5);
    graph.run();
    graph.run();
}

#[test]
#[should_panic(expected = "read before it was resolved")]
fn building_with_an_unresolved_once_value_panics() {
    let (ctx, _) = device_with(0);
    let spec = meta_node(
        MetaSource::Once(Resolvable::new()),
        MetaSource::Once(Resolvable::fixed(0.5)),
    );
    let _ = Graph::build(&ctx, &[spec]);
}

#[test]
fn a_built_graph_keeps_its_buffers_in_use() {
    let (ctx, ids) = device_with(2);
    let graph = Graph::build(&ctx, &[copy_node(ids[0], ids[1])]).unwrap();
    let unused = |id| crate::core::rules::check_unused(ctx.table(), id, ctx.device_id());
    assert!(unused(ids[0]).is_err());
    drop(graph);
    assert!(unused(ids[0]).is_ok());
}
