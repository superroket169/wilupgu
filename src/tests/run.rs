use super::*;
use crate::backend::dtype::{DataKind, DataType, F16, F32};
use crate::backend::shader::{
    BindingRole, MetaField, MetaType, Native, NativeBinding, Shader, ShaderBinary, Signature,
    Workgroups,
};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::{Arg, DeviceConfig, MetaValue};
use crate::tools::plan::call::{CallDesc, CallSpec};
use crate::tools::plan::deferred::{Dynamic, Resolvable};
use crate::tools::plan::node::{MetaSource, NodeDesc, NodeSpec};
use crate::tools::plan::run::Run;
use crate::tools::plan::tensor::{TensorDesc, TensorInit, TensorSpec};

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

static SCALE_LIKE: Signature = Signature {
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
};

fn scale_like() -> Shader {
    Shader::new(&SCALE_LIKE, ShaderBinary::Native(Native(scale)))
}

static ADD_LIKE: Signature = Signature {
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
};

fn add_like() -> Shader {
    Shader::new(&ADD_LIKE, ShaderBinary::Native(Native(add)))
}

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

fn call(shader: Shader, meta: Vec<MetaSource>, tensors: Vec<GlobalId<TensorSpec>>) -> CallSpec {
    let node = NodeSpec::new(NodeDesc {
        shader,
        meta,
        workgroups: one_wg(),
    })
    .unwrap();
    CallSpec::new(CallDesc { node, tensors }).unwrap()
}

fn scale_call(x: GlobalId<TensorSpec>, n: u32, factor: MetaSource) -> CallSpec {
    call(
        scale_like(),
        vec![MetaSource::Uint(Resolvable::fixed(n)), factor],
        vec![x],
    )
}

fn add_call(acc: GlobalId<TensorSpec>, x: GlobalId<TensorSpec>, n: u32) -> CallSpec {
    call(
        add_like(),
        vec![MetaSource::Uint(Resolvable::fixed(n))],
        vec![acc, x],
    )
}

fn fixed(v: f32) -> MetaSource {
    MetaSource::Float(Resolvable::fixed(v))
}

fn plan(tensors: Vec<TensorSpec>, calls: Vec<CallSpec>) -> RunSpec {
    RunSpec::new(RunDesc { tensors, calls }).unwrap()
}

fn values(r: &Run<'_, CpuBackend>, id: GlobalId<TensorSpec>) -> Vec<f32> {
    r.tensor(id).unwrap().download_vec::<F32>().unwrap()
}

// x doubles, then y += x.
fn double_then_add() -> (RunSpec, GlobalId<TensorSpec>, GlobalId<TensorSpec>) {
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
    assert!(RunSpec::new(RunDesc {
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
    assert!(RunSpec::new(RunDesc {
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
    let mut run = plan.build(&dev).unwrap();
    dev.scope(|s| {
        let r = run.bind(s)?;
        r.execute()?;
        r.execute()
    })
    .unwrap();
    assert_eq!(values(&run, x), vec![4.0, 8.0]);
    assert_eq!(values(&run, y), vec![16.0, 32.0]);
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
    let mut run = plan(vec![x], calls).build(&dev).unwrap();
    dev.scope(|s| run.bind(s)?.execute()).unwrap();
    assert_eq!(values(&run, x_id), vec![6.0]);
}

#[test]
fn apply_dynamics_sets_every_node_using_a_dynamic() {
    let (x, y) = (f32_tensor(vec![1.0]), f32_tensor(vec![2.0]));
    let (x_id, y_id) = (x.id(), y.id());
    let factor = Dynamic::new();
    let sx = scale_call(x_id, 1, MetaSource::FloatPerRun(factor.clone()));
    let sy = scale_call(y_id, 1, MetaSource::FloatPerRun(factor.clone()));
    let dev = device();
    let mut run = plan(vec![x, y], vec![sx, sy]).build(&dev).unwrap();

    dev.scope(|s| {
        let r = run.bind(s)?;
        assert!(r.apply_dynamics().is_err(), "never set");
        factor.set(3.0);
        r.apply_dynamics()?;
        r.execute()?;
        r.execute()?; // still 3: run doesn't read the Dynamic
        r.apply_dynamics()?; // the same value again is fine
        factor.set(10.0);
        r.apply_dynamics()?;
        r.execute()
    })
    .unwrap();
    assert_eq!(values(&run, x_id), vec![90.0]);
    assert_eq!(values(&run, y_id), vec![180.0]);
    let sx = run.order()[0];
    assert_eq!(
        run.call(sx).unwrap().node().meta()[1],
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
    let mut run = plan(vec![x], vec![sa, sb]).build(&dev).unwrap();

    a.set(2.0);
    dev.scope(|s| {
        let r = run.bind(s)?;
        assert!(r.apply_dynamics().is_err(), "b was never set");
        b.set(5.0);
        r.apply_dynamics()?;
        r.execute()
    })
    .unwrap();
    assert_eq!(values(&run, x_id), vec![10.0]);
}

#[test]
fn the_scope_reaches_the_runs_tensors_through_slots() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut run = plan.build(&dev).unwrap();
    let mut z = dev.tensor_from(&F32::wrap(vec![0.0, 0.0])).unwrap();
    let mut add_z = dev
        .node(
            &add_like(),
            &[MetaValue::Uint(2)],
            Workgroups { x: 1, y: 1, z: 1 },
        )
        .unwrap();
    let seen = dev
        .scope(|s| {
            let r = run.bind(s)?;
            let (z, add_z) = (s.bind(&mut z)?, s.bind_node(&mut add_z)?);
            s.upload(r.tensor(x)?, &F32::wrap(vec![3.0, 4.0]))?;
            r.execute()?;
            // the scope's own node reads the run's output
            s.execute(add_z, &[Arg::Write(z), Arg::Read(r.tensor(y)?)])?;
            Ok::<_, Error>(s.download(r.tensor(x)?))
        })
        .unwrap();
    assert_eq!(F32::unwrap(seen), Some(vec![6.0, 8.0]));
    assert_eq!(z.download_vec::<F32>(), Ok(vec![16.0, 28.0]));
}

#[test]
fn execute_call_runs_calls_in_any_order() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut run = plan.build(&dev).unwrap();
    let [double, add] = run.order()[..] else {
        panic!("two nodes")
    };
    dev.scope(|s| {
        let r = run.bind(s)?;
        r.execute_call(add)?; // y += x, before x doubles
        r.execute_call(double)?;
        r.execute_call(double)?;
        assert!(r.execute_call(stranger_call()).is_err());
        Ok::<_, Error>(())
    })
    .unwrap();
    assert_eq!(values(&run, y), vec![11.0, 22.0]);
    assert_eq!(values(&run, x), vec![4.0, 8.0]);
}

fn stranger_call() -> GlobalId<CallSpec> {
    scale_call(f32_tensor(vec![0.0]).id(), 1, fixed(1.0)).id()
}

#[test]
fn edit_lends_tensors_and_sets_meta() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut run = plan.build(&dev).unwrap();
    let double = run.order()[0];

    let old = run
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
        run.call(double).unwrap().node().meta()[1],
        MetaValue::Float(5.0)
    );

    dev.scope(|s| run.bind(s)?.execute()).unwrap();
    assert_eq!(values(&run, x), vec![5.0, 5.0]);
    assert_eq!(values(&run, y), vec![5.0, 5.0]);
}

#[test]
fn edit_fails_if_a_node_of_another_shader_is_left() {
    let (plan, _, _) = double_then_add();
    let dev = device();
    let mut run = plan.build(&dev).unwrap();
    let double = run.order()[0];
    let result = run.edit(|e| {
        *e.node(double)? = dev.node(
            &add_like(),
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
    let mut run = plan.build(&dev).unwrap();
    let result = run.edit(|e| {
        *e.tensor(x)? = other.tensor_from(&F32::wrap(vec![0.0, 0.0]))?;
        Ok(())
    });
    assert!(result.is_err());

    let mut run = double_then_add().0.build(&dev).unwrap();
    let double = run.order()[0];
    let result = run.edit(|e| {
        *e.node(double)? = other.node(
            &scale_like(),
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
    let mut run = plan(vec![x], vec![scale_call(x_id, 1, fixed(2.0))])
        .build(&dev)
        .unwrap();
    let scale = run.order()[0];
    // the same slots, but meta [Uint, Uint] where the call's sources are [Uint, Float]
    let result = run.edit(|e| {
        *e.node(scale)? = dev.node(
            &scale_uint(),
            &[MetaValue::Uint(1), MetaValue::Uint(2)],
            Workgroups { x: 1, y: 1, z: 1 },
        )?;
        Ok(())
    });
    assert!(result.is_err());
}

static SCALE_UINT: Signature = Signature {
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
};

fn scale_uint() -> Shader {
    Shader::new(&SCALE_UINT, ShaderBinary::Native(Native(scale)))
}

#[test]
fn edit_fails_if_a_tensor_of_another_kind_is_left() {
    let (plan, _, y) = double_then_add();
    let dev = device();
    let mut run = plan.build(&dev).unwrap();
    let result = run.edit(|e| {
        *e.tensor(y)? = dev.alloc(DataKind::F16, 2)?;
        Ok(())
    });
    assert!(result.is_err());
}

#[test]
fn reads_reach_everything_in_a_run() {
    let (plan, x, _) = double_then_add();
    let dev = device();
    let run = plan.build(&dev).unwrap();
    let double = run.order()[0];
    assert_eq!(
        run.call(double).unwrap().node().signature().name,
        "scale_like"
    );
    assert_eq!(
        run.call(double).unwrap().node().workgroups(),
        Workgroups { x: 1, y: 1, z: 1 }
    );
    assert_eq!(run.tensor(x).unwrap().len(), 2);
    assert!(run.tensor(f32_tensor(vec![0.0]).id()).is_err());
}

// Puts a 1-element tensor in place of `id`, shorter than its call needs.
fn shorten(run: &mut Run<'_, CpuBackend>, id: GlobalId<TensorSpec>) {
    let dev = run.device();
    run.edit(|edit| {
        let short = dev.alloc(DataKind::F32, 1)?;
        edit.tensor(id)?.replace(short).map_err(|(_, e)| e)?;
        Ok(())
    })
    .unwrap();
}

#[test]
fn run_fails_on_a_tensor_shortened_by_edit() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let dev = device();
    let mut run = plan(vec![x], vec![scale_call(x_id, 2, fixed(2.0))])
        .build(&dev)
        .unwrap();
    shorten(&mut run, x_id);
    assert!(dev.scope(|s| run.bind(s)?.execute()).is_err());
}

#[test]
fn captured_replays_with_the_applied_dynamics() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let factor = Dynamic::new();
    let node = scale_call(x_id, 2, MetaSource::FloatPerRun(factor.clone()));
    let dev = device();
    let run = plan(vec![x], vec![node]).build(&dev).unwrap();
    let mut captured = run.capture().map_err(|(_, e)| e).unwrap();
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
        Ok::<_, Error>(())
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
fn release_gives_back_a_run_that_runs() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let captured = plan
        .build(&dev)
        .unwrap()
        .capture()
        .map_err(|(_, e)| e)
        .unwrap();
    let mut run = captured.release();
    dev.scope(|s| run.bind(s)?.execute()).unwrap();
    assert_eq!(values(&run, x), vec![2.0, 4.0]);
    assert_eq!(values(&run, y), vec![12.0, 24.0]);
}

#[test]
fn a_failed_capture_gives_the_run_back() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let dev = device();
    let mut run = plan(vec![x], vec![scale_call(x_id, 2, fixed(2.0))])
        .build(&dev)
        .unwrap();
    shorten(&mut run, x_id);
    let Err((run, _)) = run.capture() else {
        panic!("a node needs more than its tensor holds")
    };
    assert_eq!(run.tensor(x_id).unwrap().len(), 1);
}

#[test]
fn new_rejects_a_written_tensor_in_two_slots() {
    let y = f32_tensor(vec![1.0]);
    let y_id = y.id();
    let err = RunSpec::new(RunDesc {
        tensors: vec![y],
        calls: vec![add_call(y_id, y_id, 1)],
    })
    .err();
    assert_eq!(
        err,
        Some(Error::Aliased {
            shader: "add_like",
            slot: 0
        })
    );
}

#[test]
fn build_rejects_a_tensor_shorter_than_its_slot_needs() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let p = plan(vec![x], vec![scale_call(x_id, 3, fixed(2.0))]);
    let err = p.build(&device()).err();
    assert_eq!(
        err,
        Some(Error::TooShort {
            shader: "scale_like",
            slot: 0,
            needs: 3,
            holds: 2
        })
    );
}

#[test]
fn a_failed_run_names_the_call() {
    let (x, y) = (f32_tensor(vec![1.0, 2.0]), f32_tensor(vec![10.0, 20.0]));
    let (x_id, y_id) = (x.id(), y.id());
    let calls = vec![scale_call(x_id, 2, fixed(2.0)), add_call(y_id, x_id, 2)];
    let dev = device();
    let mut run = plan(vec![x, y], calls).build(&dev).unwrap();
    shorten(&mut run, y_id);
    let err = dev.scope(|s| run.bind(s)?.execute()).err();
    assert_eq!(
        err,
        Some(Error::InCall {
            index: 1,
            error: Box::new(Error::TooShort {
                shader: "add_like",
                slot: 0,
                needs: 2,
                holds: 1
            })
        })
    );
    assert_eq!(values(&run, x_id), vec![2.0, 4.0]);
}

#[test]
fn build_hollow_makes_every_tensor_hollow() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let run = plan.build_hollow(&dev).unwrap();
    assert!(run.contains_hollow());
    assert!(run.tensors().all(|(_, t)| t.is_hollow()));
    let x_hollow = run.any_tensor(x).unwrap();
    assert_eq!((x_hollow.kind(), x_hollow.len()), (DataKind::F32, 2));
    assert_eq!(run.tensor(y).err(), Some(Error::Hollow));
    assert_eq!(run.order().len(), 2);
    assert!(!plan.build(&dev).unwrap().contains_hollow());
}

#[test]
fn build_hollow_can_be_called_again() {
    let (plan, _, _) = double_then_add();
    let dev = device();
    let a = plan.build_hollow(&dev).unwrap();
    let b = plan.build_hollow(&dev).unwrap();
    assert_eq!(a.order(), b.order());
}

#[test]
fn a_run_with_a_hollow_tensor_does_not_bind() {
    let (plan, _, _) = double_then_add();
    let dev = device();
    let mut run = plan.build_hollow(&dev).unwrap();
    let err = dev.scope(|s| run.bind(s).map(|_| ())).err();
    assert_eq!(err, Some(Error::Hollow));
}

#[test]
fn edit_does_not_lend_a_hollow_tensor() {
    let (plan, x, _) = double_then_add();
    let dev = device();
    let mut run = plan.build_hollow(&dev).unwrap();
    let err = run.edit(|e| e.tensor(x).map(|_| ())).err();
    assert_eq!(err, Some(Error::Hollow));
}

#[test]
fn build_hollow_rejects_a_tensor_shorter_than_its_slot_needs() {
    let x = f32_tensor(vec![1.0, 2.0]);
    let x_id = x.id();
    let p = plan(vec![x], vec![scale_call(x_id, 3, fixed(2.0))]);
    assert!(matches!(
        p.build_hollow(&device()).err(),
        Some(Error::TooShort { .. })
    ));
}

#[test]
fn fill_from_moves_a_live_tensor_into_a_hollow_place() {
    let (plan, x, y) = double_then_add();
    let dev = device();
    let mut live = plan.build(&dev).unwrap();
    let mut hollow = plan.build_hollow(&dev).unwrap();
    hollow.fill_from(&mut live, x).unwrap();
    hollow.fill_from(&mut live, y).unwrap();
    assert!(!hollow.contains_hollow());
    assert!(live.tensors().all(|(_, t)| t.is_hollow()));
    assert_eq!(values(&hollow, x), vec![1.0, 2.0]);
    dev.scope(|s| hollow.bind(s)?.execute()).unwrap();
    assert_eq!(values(&hollow, y), vec![12.0, 24.0]);

    live.fill_from(&mut hollow, x).unwrap();
    assert_eq!(values(&live, x), vec![2.0, 4.0]);
    assert!(hollow.any_tensor(x).unwrap().is_hollow());
}

#[test]
fn fill_from_needs_a_hollow_place_and_a_live_tensor() {
    let (plan, x, _) = double_then_add();
    let dev = device();
    let mut a = plan.build(&dev).unwrap();
    let mut b = plan.build(&dev).unwrap();
    assert_eq!(a.fill_from(&mut b, x), Err(Error::NotHollow));
    let mut c = plan.build_hollow(&dev).unwrap();
    let mut d = plan.build_hollow(&dev).unwrap();
    assert_eq!(c.fill_from(&mut d, x), Err(Error::Hollow));
    let stranger = f32_tensor(vec![0.0]).id();
    assert!(matches!(
        c.fill_from(&mut a, stranger),
        Err(Error::NotInPlan { .. })
    ));
    assert!(!a.contains_hollow());
}

#[test]
fn fill_from_rejects_another_device_or_length() {
    let (plan, x, _) = double_then_add();
    let (dev, other) = (device(), device());
    let mut hollow = plan.build_hollow(&dev).unwrap();
    let mut elsewhere = plan.build(&other).unwrap();
    assert!(matches!(
        hollow.fill_from(&mut elsewhere, x),
        Err(Error::OtherDevice { .. })
    ));
    let mut short = plan.build(&dev).unwrap();
    shorten(&mut short, x);
    assert!(matches!(
        hollow.fill_from(&mut short, x),
        Err(Error::LenMismatch { .. })
    ));
    assert!(!short.any_tensor(x).unwrap().is_hollow());
}
