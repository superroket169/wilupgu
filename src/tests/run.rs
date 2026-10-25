use super::*;
use crate::backend::dtype::{DataKind, DataType, F16, F32};
use crate::backend::shader::{
    MetaField, MetaType, NativeBinding, NativeCode, Shader, ShaderCode, Workgroups,
};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::DeviceConfig;
use crate::safe::{Arg, MetaValue};
use crate::tools::plan::deferred::{Dynamic, Resolvable};
use crate::tools::plan::node::MetaSource;
use crate::tools::plan::node::{Binding, NodeDesc};
use crate::tools::plan::runnable::Runnable;
use crate::tools::plan::tensor::{TensorDesc, TensorInit};

fn scale(meta: &[u32], bindings: &mut [NativeBinding]) {
    let [NativeBinding::Write(x)] = bindings else {
        panic!("scale_like takes one written slot")
    };
    let x = bytemuck::cast_slice_mut::<u8, f32>(x);
    let factor = f32::from_bits(meta[1]);
    for v in &mut x[..meta[0] as usize] {
        *v *= factor;
    }
}

fn add(meta: &[u32], bindings: &mut [NativeBinding]) {
    let [NativeBinding::Write(acc), NativeBinding::Read(x)] = bindings else {
        panic!("add_like takes a written and a read slot")
    };
    let acc = bytemuck::cast_slice_mut::<u8, f32>(acc);
    let x = bytemuck::cast_slice::<u8, f32>(x);
    for i in 0..meta[0] as usize {
        acc[i] += x[i];
    }
}

static SCALE_LIKE: Shader = Shader {
    name: "scale_like",
    meta: &[
        MetaField {
            name: "n",
            ty: MetaType::Uint,
        },
        MetaField {
            name: "factor",
            ty: MetaType::Float,
        },
    ],
    layout: &[BindingRole::InOut(DataKind::F32)],
    min_lens: |m, _, l| l.fill(m[0] as usize),
    workgroup_size: [256, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(scale)),
        ..ShaderCode::NONE
    },
};

static ADD_LIKE: Shader = Shader {
    name: "add_like",
    meta: &[MetaField {
        name: "n",
        ty: MetaType::Uint,
    }],
    layout: &[
        BindingRole::Accumulate(DataKind::F32),
        BindingRole::Input(DataKind::F32),
    ],
    min_lens: |m, _, l| l.fill(m[0] as usize),
    workgroup_size: [256, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(add)),
        ..ShaderCode::NONE
    },
};

fn device() -> ComputeDevice<CpuBackend> {
    ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    )
}

fn f32_tensor(data: Vec<f32>) -> TensorSpec {
    TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len: Resolvable::fixed(data.len()),
        init: TensorInit::Upload(F32::wrap(data)),
    })
    .unwrap()
}

fn one_wg() -> Resolvable<Workgroups> {
    Resolvable::fixed(Workgroups { x: 1, y: 1, z: 1 })
}

fn scale_node(x: TensorId, n: u32, factor: MetaSource) -> NodeSpec {
    NodeSpec::new(NodeDesc {
        shader: &SCALE_LIKE,
        meta: vec![MetaSource::Uint(Resolvable::fixed(n)), factor],
        bindings: vec![Binding::new(1, x)],
        workgroups: one_wg(),
    })
    .unwrap()
}

fn add_node(acc: TensorId, x: TensorId, n: u32) -> NodeSpec {
    NodeSpec::new(NodeDesc {
        shader: &ADD_LIKE,
        meta: vec![MetaSource::Uint(Resolvable::fixed(n))],
        bindings: vec![Binding::new(1, acc), Binding::new(2, x)],
        workgroups: one_wg(),
    })
    .unwrap()
}

fn fixed(v: f32) -> MetaSource {
    MetaSource::Float(Resolvable::fixed(v))
}

fn part(nodes: &[&NodeSpec]) -> RunPart {
    RunPart {
        nodes: nodes.iter().map(|n| n.id()).collect(),
    }
}

fn plan(tensors: Vec<TensorSpec>, nodes: Vec<NodeSpec>, parts: Vec<RunPart>) -> RunPlan {
    RunPlan::new(PlanDesc {
        tensors,
        nodes,
        parts,
    })
    .unwrap()
}

fn values(r: &Runnable<'_, CpuBackend>, id: TensorId) -> Vec<f32> {
    r.tensor(id).unwrap().download_vec::<F32>().unwrap()
}

// x doubles in part 0, then y += x in part 1.
fn double_then_add() -> (RunPlan, TensorId, TensorId) {
    let (x, y) = (f32_tensor(vec![1.0, 2.0]), f32_tensor(vec![10.0, 20.0]));
    let (x_id, y_id) = (x.id(), y.id());
    let double = scale_node(x_id, 2, fixed(2.0));
    let add = add_node(y_id, x_id, 2);
    let parts = vec![part(&[&double]), part(&[&add])];
    (plan(vec![x, y], vec![add, double], parts), x_id, y_id)
}

#[test]
fn new_rejects_a_binding_outside_the_plan_or_of_another_kind() {
    let x = f32_tensor(vec![1.0]);
    let node = scale_node(x.id(), 1, fixed(2.0));
    let parts = vec![part(&[&node])];
    assert!(RunPlan::new(PlanDesc {
        tensors: vec![],
        nodes: vec![node],
        parts,
    })
    .is_err());

    let half = TensorSpec::new(TensorDesc {
        kind: DataKind::F16,
        len: Resolvable::fixed(1),
        init: TensorInit::Upload(F16::wrap(vec![half::f16::ONE])),
    })
    .unwrap();
    let node = scale_node(half.id(), 1, fixed(2.0));
    let parts = vec![part(&[&node])];
    assert!(RunPlan::new(PlanDesc {
        tensors: vec![half],
        nodes: vec![node],
        parts,
    })
    .is_err());
}

#[test]
fn new_rejects_an_unknown_node_in_a_part_and_a_node_in_no_part() {
    let x = f32_tensor(vec![1.0]);
    let a = scale_node(x.id(), 1, fixed(2.0));
    let stranger = scale_node(x.id(), 1, fixed(2.0));
    let parts = vec![part(&[&a, &stranger])];
    assert!(RunPlan::new(PlanDesc {
        tensors: vec![x],
        nodes: vec![a],
        parts,
    })
    .is_err());

    let x = f32_tensor(vec![1.0]);
    let a = scale_node(x.id(), 1, fixed(2.0));
    let idle = scale_node(x.id(), 1, fixed(2.0));
    let parts = vec![part(&[&a])];
    assert!(RunPlan::new(PlanDesc {
        tensors: vec![x],
        nodes: vec![a, idle],
        parts,
    })
    .is_err());
}

#[test]
fn ready_waits_for_every_spec() {
    let len = Resolvable::new();
    let resolver = len.resolver().unwrap();
    let x = TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len,
        init: TensorInit::Zero,
    })
    .unwrap();
    let node = scale_node(x.id(), 2, fixed(2.0));
    let parts = vec![part(&[&node])];
    let plan = plan(vec![x], vec![node], parts);
    assert!(plan.ready().is_err());
    assert!(plan.build(&device()).is_err());
    resolver.resolve(2);
    assert!(plan.ready().is_ok());
}

#[test]
fn runs_in_one_scope_follow_the_parts_in_order() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    dev.scope(|s| {
        let r = runnable.bind(s)?;
        r.run()?;
        r.run()
    })
    .unwrap();
    assert_eq!(values(&runnable, x), vec![4.0, 8.0]);
    assert_eq!(values(&runnable, y), vec![16.0, 32.0]);
}

#[test]
fn a_node_can_run_twice_in_a_part() {
    let x = f32_tensor(vec![1.0]);
    let x_id = x.id();
    let double = scale_node(x_id, 1, fixed(2.0));
    let parts = vec![part(&[&double, &double])];
    let dev = device();
    let mut runnable = plan(vec![x], vec![double], parts).build(&dev).unwrap();
    dev.scope(|s| runnable.bind(s)?.run()).unwrap();
    assert_eq!(values(&runnable, x_id), vec![4.0]);
}

#[test]
fn a_shared_dynamic_is_read_once_per_run() {
    let (x, y) = (f32_tensor(vec![1.0]), f32_tensor(vec![2.0]));
    let (x_id, y_id) = (x.id(), y.id());
    let factor = Dynamic::new();
    let sx = scale_node(x_id, 1, MetaSource::FloatPerRun(factor.clone()));
    let sy = scale_node(y_id, 1, MetaSource::FloatPerRun(factor.clone()));
    let parts = vec![part(&[&sx, &sy])];
    let dev = device();
    let mut runnable = plan(vec![x, y], vec![sx, sy], parts).build(&dev).unwrap();

    dev.scope(|s| {
        let r = runnable.bind(s)?;
        assert!(r.run().is_err(), "never set");
        factor.set(3.0);
        r.run()?;
        assert!(r.run().is_err(), "not set again since the last run");
        factor.set(10.0);
        r.run()
    })
    .unwrap();
    assert_eq!(values(&runnable, x_id), vec![30.0]);
    assert_eq!(values(&runnable, y_id), vec![60.0]);
    let sx = runnable.parts()[0].nodes[0];
    assert_eq!(
        runnable.node(sx).unwrap().meta()[1],
        MetaValue::Float(10.0),
        "a node's meta shows the last run's Dynamic"
    );
}

#[test]
fn a_failed_run_reads_no_dynamic() {
    let x = f32_tensor(vec![1.0]);
    let x_id = x.id();
    let (a, b) = (Dynamic::new(), Dynamic::new());
    let sa = scale_node(x_id, 1, MetaSource::FloatPerRun(a.clone()));
    let sb = scale_node(x_id, 1, MetaSource::FloatPerRun(b.clone()));
    let parts = vec![part(&[&sa, &sb])];
    let dev = device();
    let mut runnable = plan(vec![x], vec![sa, sb], parts).build(&dev).unwrap();

    a.set(2.0);
    dev.scope(|s| {
        let r = runnable.bind(s)?;
        assert!(r.run().is_err(), "b was never set");
        b.set(5.0);
        r.run()
    })
    .unwrap();
    assert_eq!(values(&runnable, x_id), vec![10.0]);
}

#[test]
fn the_scope_reaches_the_runnables_tensors_through_slots() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    let mut z = dev.tensor_from(&F32::wrap(vec![0.0, 0.0])).unwrap();
    let mut add_z = dev
        .node(
            &ADD_LIKE,
            &[MetaValue::Uint(2)],
            Workgroups { x: 1, y: 1, z: 1 },
        )
        .unwrap();
    let seen = dev
        .scope(|s| {
            let r = runnable.bind(s)?;
            let (z, add_z) = (s.bind(&mut z)?, s.bind_node(&mut add_z)?);
            s.upload(r.tensor(x)?, &F32::wrap(vec![3.0, 4.0]))?;
            r.run()?;
            // the scope's own node reads the runnable's output
            s.run(add_z, &[Arg::Write(z), Arg::Read(r.tensor(y)?)])?;
            Ok::<_, String>(s.download(r.tensor(x)?))
        })
        .unwrap();
    assert_eq!(F32::unwrap(seen), Some(vec![6.0, 8.0]));
    assert_eq!(z.download_vec::<F32>(), Ok(vec![16.0, 28.0]));
}

#[test]
fn stream_runs_parts_in_the_order_it_picks() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    dev.scope(|s| {
        let r = runnable.bind(s)?;
        r.stream(|st| {
            assert_eq!(st.parts(), 2);
            st.run_part(1)?; // y += x, before x doubles
            st.run_part(0)?;
            st.run_part(0)?;
            assert!(st.run_part(2).is_err());
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(values(&runnable, y), vec![11.0, 22.0]);
    assert_eq!(values(&runnable, x), vec![4.0, 8.0]);
}

#[test]
fn edit_lends_tensors_and_sets_meta() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    let double = runnable.parts()[0].nodes[0];

    let old = runnable
        .edit(|e| {
            e.tensor(x)?.upload(&F32::wrap(vec![1.0, 1.0]))?;
            e.node(double)?
                .update_meta(&[MetaValue::Uint(2), MetaValue::Float(5.0)])?;
            let fresh = dev.tensor_from(&F32::wrap(vec![0.0, 0.0]))?;
            Ok(std::mem::replace(e.tensor(y)?, fresh))
        })
        .unwrap();
    assert_eq!(old.download_vec::<F32>(), Ok(vec![10.0, 20.0]));
    assert_eq!(
        runnable.node(double).unwrap().meta()[1],
        MetaValue::Float(5.0)
    );

    dev.scope(|s| runnable.bind(s)?.run()).unwrap();
    assert_eq!(values(&runnable, x), vec![5.0, 5.0]);
    assert_eq!(values(&runnable, y), vec![5.0, 5.0]);
}

#[test]
fn edit_fails_if_a_node_of_another_shader_is_left() {
    let (plan, _, _) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    let double = runnable.parts()[0].nodes[0];
    let result = runnable.edit(|e| {
        *e.node(double)? = dev.node(
            &ADD_LIKE,
            &[MetaValue::Uint(2)],
            Workgroups { x: 1, y: 1, z: 1 },
        )?;
        Ok(())
    });
    assert!(result.is_err());
}

#[test]
fn edit_fails_if_a_tensor_of_another_kind_is_left() {
    let (plan, _, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    let result = runnable.edit(|e| {
        *e.tensor(y)? = dev.alloc(DataKind::F16, 2)?;
        Ok(())
    });
    assert!(result.is_err());
}

#[test]
fn reads_reach_every_part_of_a_runnable() {
    let (plan, x, _) = double_then_add();
    let dev = device();
    let runnable = plan.build(&dev).unwrap();
    let double = runnable.parts()[0].nodes[0];
    assert_eq!(runnable.node(double).unwrap().shader().name, "scale_like");
    assert_eq!(
        runnable.node(double).unwrap().workgroups(),
        Workgroups { x: 1, y: 1, z: 1 }
    );
    assert_eq!(runnable.tensor(x).unwrap().len(), 2);
    assert!(runnable.tensor(f32_tensor(vec![0.0]).id()).is_err());
}

#[test]
fn run_fails_on_a_tensor_shorter_than_the_node_needs() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let short = scale_node(x.id(), 4, fixed(2.0));
    let parts = vec![part(&[&short])];
    let dev = device();
    let mut runnable = plan(vec![x], vec![short], parts).build(&dev).unwrap();
    assert!(dev.scope(|s| runnable.bind(s)?.run()).is_err());
}

#[test]
fn captured_replays_with_this_runs_dynamics() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let factor = Dynamic::new();
    let node = scale_node(x_id, 2, MetaSource::FloatPerRun(factor.clone()));
    let parts = vec![part(&[&node])];
    let dev = device();
    let runnable = plan(vec![x], vec![node], parts).build(&dev).unwrap();
    let mut captured = runnable.capture().map_err(|(_, e)| e).unwrap();
    assert_eq!(
        captured.tensor(x_id).unwrap().download_vec::<F32>(),
        Ok(vec![1.0, 2.0]),
        "recording runs nothing"
    );

    dev.scope(|s| {
        let r = captured.bind(s)?;
        factor.set(2.0);
        r.replay()?;
        factor.set(3.0);
        r.replay()?;
        assert!(r.replay().is_err(), "not set again since the last replay");
        Ok::<_, String>(())
    })
    .unwrap();
    assert_eq!(
        captured.tensor(x_id).unwrap().download_vec::<F32>(),
        Ok(vec![6.0, 12.0])
    );

    captured.upload(x_id, &F32::wrap(vec![1.0, 1.0])).unwrap();
    factor.set(4.0);
    dev.scope(|s| captured.bind(s)?.replay()).unwrap();
    assert_eq!(
        captured.tensor(x_id).unwrap().download_vec::<F32>(),
        Ok(vec![4.0, 4.0])
    );
}

#[test]
fn release_gives_back_a_runnable_that_runs() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let captured = plan
        .build(&dev)
        .unwrap()
        .capture()
        .map_err(|(_, e)| e)
        .unwrap();
    let mut runnable = captured.release();
    dev.scope(|s| runnable.bind(s)?.run()).unwrap();
    assert_eq!(values(&runnable, x), vec![2.0, 4.0]);
    assert_eq!(values(&runnable, y), vec![12.0, 24.0]);
}

#[test]
fn a_failed_capture_gives_the_runnable_back() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let short = scale_node(x_id, 4, fixed(2.0));
    let parts = vec![part(&[&short])];
    let dev = device();
    let runnable = plan(vec![x], vec![short], parts).build(&dev).unwrap();
    let Err((runnable, _)) = runnable.capture() else {
        panic!("a node needs more than its tensor holds")
    };
    assert_eq!(values(&runnable, x_id), vec![1.0, 2.0]);
}
