use super::*;
use crate::backend::dtype::{DataKind, DataType, F16, F32};
use crate::backend::shader::{
    MetaField, MetaType, NativeBinding, NativeCode, Shader, ShaderCode, Workgroups,
};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::{Arg, DeviceConfig, MetaValue};
use crate::tools::plan::call::{CallDesc, CallId, CallSpec};
use crate::tools::plan::deferred::{Dynamic, Resolvable};
use crate::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use crate::tools::plan::runnable::Runnable;
use crate::tools::plan::tensor::{TensorDesc, TensorId, TensorInit, TensorSpec};

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

fn call(shader: &'static Shader, meta: Vec<MetaSource>, tensors: Vec<TensorId>) -> CallSpec {
    let node = NodeSpec::new(NodeDesc {
        shader,
        meta,
        workgroups: one_wg(),
    })
    .unwrap();
    CallSpec::new(CallDesc { node, tensors }).unwrap()
}

fn scale_call(x: TensorId, n: u32, factor: MetaSource) -> CallSpec {
    call(
        &SCALE_LIKE,
        vec![MetaSource::Uint(Resolvable::fixed(n)), factor],
        vec![x],
    )
}

fn add_call(acc: TensorId, x: TensorId, n: u32) -> CallSpec {
    call(
        &ADD_LIKE,
        vec![MetaSource::Uint(Resolvable::fixed(n))],
        vec![acc, x],
    )
}

fn fixed(v: f32) -> MetaSource {
    MetaSource::Float(Resolvable::fixed(v))
}

fn plan(tensors: Vec<TensorSpec>, calls: Vec<CallSpec>) -> RunPlan {
    RunPlan::new(PlanDesc { tensors, calls }).unwrap()
}

fn values(r: &Runnable<'_, CpuBackend>, id: TensorId) -> Vec<f32> {
    r.tensor(id).unwrap().download_vec::<F32>().unwrap()
}

// x doubles, then y += x.
fn double_then_add() -> (RunPlan, TensorId, TensorId) {
    let (x, y) = (f32_tensor(vec![1.0, 2.0]), f32_tensor(vec![10.0, 20.0]));
    let (x_id, y_id) = (x.id(), y.id());
    let double = scale_call(x_id, 2, fixed(2.0));
    let add = add_call(y_id, x_id, 2);
    (plan(vec![x, y], vec![double, add]), x_id, y_id)
}

#[test]
fn new_rejects_a_binding_outside_the_plan_or_of_another_kind() {
    let x = f32_tensor(vec![1.0]);
    let node = scale_call(x.id(), 1, fixed(2.0));
    assert!(RunPlan::new(PlanDesc {
        tensors: vec![],
        calls: vec![node],
    })
    .is_err());

    let half = TensorSpec::new(TensorDesc {
        kind: DataKind::F16,
        len: Resolvable::fixed(1),
        init: TensorInit::Upload(F16::wrap(vec![half::f16::ONE])),
    })
    .unwrap();
    let node = scale_call(half.id(), 1, fixed(2.0));
    assert!(RunPlan::new(PlanDesc {
        tensors: vec![half],
        calls: vec![node],
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
    let node = scale_call(x.id(), 2, fixed(2.0));
    let plan = plan(vec![x], vec![node]);
    assert!(plan.ready().is_err());
    assert!(plan.build(&device()).is_err());
    resolver.resolve(2);
    assert!(plan.ready().is_ok());
}

#[test]
fn runs_in_one_scope_follow_the_order() {
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
fn calls_of_one_shader_run_in_order() {
    let x = f32_tensor(vec![1.0]);
    let x_id = x.id();
    let calls = vec![
        scale_call(x_id, 1, fixed(2.0)),
        scale_call(x_id, 1, fixed(3.0)),
    ];
    let dev = device();
    let mut runnable = plan(vec![x], calls).build(&dev).unwrap();
    dev.scope(|s| runnable.bind(s)?.run()).unwrap();
    assert_eq!(values(&runnable, x_id), vec![6.0]);
}

#[test]
fn apply_dynamics_sets_every_node_using_a_dynamic() {
    let (x, y) = (f32_tensor(vec![1.0]), f32_tensor(vec![2.0]));
    let (x_id, y_id) = (x.id(), y.id());
    let factor = Dynamic::new();
    let sx = scale_call(x_id, 1, MetaSource::FloatPerRun(factor.clone()));
    let sy = scale_call(y_id, 1, MetaSource::FloatPerRun(factor.clone()));
    let dev = device();
    let mut runnable = plan(vec![x, y], vec![sx, sy]).build(&dev).unwrap();

    dev.scope(|s| {
        let r = runnable.bind(s)?;
        assert!(r.apply_dynamics().is_err(), "never set");
        factor.set(3.0);
        r.apply_dynamics()?;
        r.run()?;
        r.run()?; // still 3: run doesn't read the Dynamic
        r.apply_dynamics()?; // the same value again is fine
        factor.set(10.0);
        r.apply_dynamics()?;
        r.run()
    })
    .unwrap();
    assert_eq!(values(&runnable, x_id), vec![90.0]);
    assert_eq!(values(&runnable, y_id), vec![180.0]);
    let sx = runnable.order()[0];
    assert_eq!(
        runnable.call(sx).unwrap().node().meta()[1],
        MetaValue::Float(10.0),
        "a node's meta shows the last applied Dynamic"
    );
}

#[test]
fn a_failed_apply_changes_no_meta() {
    let x = f32_tensor(vec![1.0]);
    let x_id = x.id();
    let (a, b) = (Dynamic::new(), Dynamic::new());
    let sa = scale_call(x_id, 1, MetaSource::FloatPerRun(a.clone()));
    let sb = scale_call(x_id, 1, MetaSource::FloatPerRun(b.clone()));
    let dev = device();
    let mut runnable = plan(vec![x], vec![sa, sb]).build(&dev).unwrap();

    a.set(2.0);
    dev.scope(|s| {
        let r = runnable.bind(s)?;
        assert!(r.apply_dynamics().is_err(), "b was never set");
        b.set(5.0);
        r.apply_dynamics()?;
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
fn run_call_runs_calls_in_any_order() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    let [double, add] = runnable.order()[..] else {
        panic!("two nodes")
    };
    dev.scope(|s| {
        let r = runnable.bind(s)?;
        r.run_call(add)?; // y += x, before x doubles
        r.run_call(double)?;
        r.run_call(double)?;
        assert!(r.run_call(stranger_call()).is_err());
        Ok::<_, String>(())
    })
    .unwrap();
    assert_eq!(values(&runnable, y), vec![11.0, 22.0]);
    assert_eq!(values(&runnable, x), vec![4.0, 8.0]);
}

fn stranger_call() -> CallId {
    scale_call(f32_tensor(vec![0.0]).id(), 1, fixed(1.0)).id()
}

#[test]
fn edit_lends_tensors_and_sets_meta() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut runnable = plan.build(&dev).unwrap();
    let double = runnable.order()[0];

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
        runnable.call(double).unwrap().node().meta()[1],
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
    let double = runnable.order()[0];
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
fn edit_fails_if_a_tensor_or_node_of_another_device_is_left() {
    let (plan, x, _) = double_then_add();
    let (dev, other) = (device(), device());
    let mut runnable = plan.build(&dev).unwrap();
    let result = runnable.edit(|e| {
        *e.tensor(x)? = other.tensor_from(&F32::wrap(vec![0.0, 0.0]))?;
        Ok(())
    });
    assert!(result.is_err());

    let mut runnable = double_then_add().0.build(&dev).unwrap();
    let double = runnable.order()[0];
    let result = runnable.edit(|e| {
        *e.node(double)? = other.node(
            &SCALE_LIKE,
            &[MetaValue::Uint(2), MetaValue::Float(2.0)],
            Workgroups { x: 1, y: 1, z: 1 },
        )?;
        Ok(())
    });
    assert!(result.is_err());
}

#[test]
fn edit_fails_if_a_node_with_other_meta_is_left() {
    let x = f32_tensor(vec![1.0]);
    let x_id = x.id();
    let dev = device();
    let mut runnable = plan(vec![x], vec![scale_call(x_id, 1, fixed(2.0))])
        .build(&dev)
        .unwrap();
    let scale = runnable.order()[0];
    // the same slots, but meta [Uint, Uint] where the call's sources are [Uint, Float]
    let result = runnable.edit(|e| {
        *e.node(scale)? = dev.node(
            &SCALE_UINT,
            &[MetaValue::Uint(1), MetaValue::Uint(2)],
            Workgroups { x: 1, y: 1, z: 1 },
        )?;
        Ok(())
    });
    assert!(result.is_err());
}

static SCALE_UINT: Shader = Shader {
    name: "scale_uint",
    meta: &[
        MetaField {
            name: "n",
            ty: MetaType::Uint,
        },
        MetaField {
            name: "factor",
            ty: MetaType::Uint,
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
fn reads_reach_everything_in_a_runnable() {
    let (plan, x, _) = double_then_add();
    let dev = device();
    let runnable = plan.build(&dev).unwrap();
    let double = runnable.order()[0];
    assert_eq!(
        runnable.call(double).unwrap().node().shader().name,
        "scale_like"
    );
    assert_eq!(
        runnable.call(double).unwrap().node().workgroups(),
        Workgroups { x: 1, y: 1, z: 1 }
    );
    assert_eq!(runnable.tensor(x).unwrap().len(), 2);
    assert!(runnable.tensor(f32_tensor(vec![0.0]).id()).is_err());
}

#[test]
fn run_fails_on_a_tensor_shorter_than_a_call_needs() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let short = scale_call(x.id(), 4, fixed(2.0));
    let dev = device();
    let mut runnable = plan(vec![x], vec![short]).build(&dev).unwrap();
    assert!(dev.scope(|s| runnable.bind(s)?.run()).is_err());
}

#[test]
fn captured_replays_with_the_applied_dynamics() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let factor = Dynamic::new();
    let node = scale_call(x_id, 2, MetaSource::FloatPerRun(factor.clone()));
    let dev = device();
    let runnable = plan(vec![x], vec![node]).build(&dev).unwrap();
    let mut captured = runnable.capture().map_err(|(_, e)| e).unwrap();
    assert_eq!(
        captured.tensor(x_id).unwrap().download_vec::<F32>(),
        Ok(vec![1.0, 2.0]),
        "recording runs nothing"
    );

    dev.scope(|s| {
        let r = captured.bind(s)?;
        factor.set(2.0);
        r.apply_dynamics()?;
        r.replay()?;
        factor.set(3.0);
        r.apply_dynamics()?;
        r.replay()?;
        Ok::<_, String>(())
    })
    .unwrap();
    assert_eq!(
        captured.tensor(x_id).unwrap().download_vec::<F32>(),
        Ok(vec![6.0, 12.0])
    );

    captured.upload(x_id, &F32::wrap(vec![1.0, 1.0])).unwrap();
    factor.set(4.0);
    dev.scope(|s| {
        let r = captured.bind(s)?;
        r.apply_dynamics()?;
        r.replay()
    })
    .unwrap();
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
    let short = scale_call(x_id, 4, fixed(2.0));
    let dev = device();
    let runnable = plan(vec![x], vec![short]).build(&dev).unwrap();
    let Err((runnable, _)) = runnable.capture() else {
        panic!("a node needs more than its tensor holds")
    };
    assert_eq!(values(&runnable, x_id), vec![1.0, 2.0]);
}
