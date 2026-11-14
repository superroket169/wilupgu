use super::*;
use crate::backend::dtype::{DataKind, DataType, F32};
use crate::backend::shader::{MetaType, Shader, ShaderBinary};
use crate::safe::{Arg, ComputeDevice, DeviceConfig, MetaValue};
use crate::tools::builtins::{PortableShader, ADD, ALL, SCALE};

// The first adapter, or `None`, with a note, without one.
fn device(submit_every: u8) -> Option<ComputeDevice<WgpuBackend>> {
    let Some(info) = WgpuBackend::choosable_devices().into_iter().next() else {
        eprintln!("no wgpu adapter; skipped");
        return None;
    };
    let wgpu = WgpuBackend::attach(info).unwrap();
    wgpu.submit_every.store(submit_every, Ordering::Relaxed);
    Some(ComputeDevice::new(wgpu, DeviceConfig::default()))
}

fn shader(builtin: &'static PortableShader) -> Shader {
    let shader = builtin.shader_for::<WgpuBackend>().unwrap();
    assert!(matches!(shader.binary(), ShaderBinary::Wgsl(_)));
    shader
}

fn scale_meta(n: u32, factor: f32) -> [MetaValue; 2] {
    [MetaValue::Uint(n), MetaValue::Float(factor)]
}

#[test]
fn choosable_devices_never_panics() {
    let _ = WgpuBackend::choosable_devices();
}

#[test]
fn every_builtin_builds() {
    let Some(dev) = device(1) else { return };
    for b in ALL {
        let meta: Vec<MetaValue> = b
            .signature
            .meta
            .iter()
            .map(|field| match field.ty {
                MetaType::Uint => MetaValue::Uint(1),
                MetaType::Float => MetaValue::Float(0.5),
            })
            .collect();
        dev.node(&shader(b), &meta, Workgroups::linear(1))
            .unwrap_or_else(|e| panic!("`{}`: {e}", b.signature.name));
    }
}

#[test]
fn a_broken_shader_fails_to_build() {
    let Some(dev) = device(1) else { return };
    let broken = Shader::new(
        &ADD.signature,
        ShaderBinary::Wgsl(crate::backend::shader::Wgsl("not wgsl".into())),
    );
    assert!(dev
        .node(&broken, &[MetaValue::Uint(3)], Workgroups::linear(1))
        .is_err());
}

#[test]
fn add_runs() {
    let Some(dev) = device(1) else { return };
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0])).unwrap();
    let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 20.0, 30.0])).unwrap();
    let mut add = dev
        .node(&shader(&ADD), &[MetaValue::Uint(3)], Workgroups::linear(1))
        .unwrap();
    dev.scope(|s| {
        let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
        s.execute(add, &[Arg::Write(x), Arg::Read(y)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn from_device_runs_on_the_callers_device() {
    let Some(info) = WgpuBackend::choosable_devices().into_iter().next() else {
        eprintln!("no wgpu adapter; skipped");
        return;
    };
    let adapter = info.adapter.clone();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let wgpu = WgpuBackend::from_device(adapter, device, queue);
    assert!(wgpu.info().label().contains("wgpu:given"));

    let dev = ComputeDevice::new(wgpu, DeviceConfig::default());
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
    let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 20.0])).unwrap();
    let mut add = dev
        .node(&shader(&ADD), &[MetaValue::Uint(2)], Workgroups::linear(1))
        .unwrap();
    dev.scope(|s| {
        let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
        s.execute(add, &[Arg::Write(x), Arg::Read(y)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![11.0, 22.0]);
}

#[test]
fn an_f16_shader_names_the_missing_feature() {
    let Some(info) = WgpuBackend::choosable_devices().into_iter().next() else {
        eprintln!("no wgpu adapter; skipped");
        return;
    };
    let adapter = info.adapter.clone();
    // no features requested, so no `SHADER_F16`
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let dev = ComputeDevice::new(
        WgpuBackend::from_device(adapter, device, queue),
        DeviceConfig::default(),
    );
    let f16 = Shader::new(
        &ADD.signature,
        ShaderBinary::Wgsl(crate::backend::shader::Wgsl(
            format!("enable f16;\n{}", ADD.wgsl.0).into(),
        )),
    );
    let err = dev
        .node(&f16, &[MetaValue::Uint(3)], Workgroups::linear(1))
        .err()
        .expect("an f16 shader builds without SHADER_F16");
    assert!(err.to_string().contains("SHADER_F16"), "{err}");
}

// Submitting after every command, after four, and only at the end.
#[test]
fn update_meta_reaches_only_the_runs_after_it() {
    for submit_every in [1, 4, 0] {
        let Some(dev) = device(submit_every) else {
            return;
        };
        let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
        let mut scale = dev
            .node(&shader(&SCALE), &scale_meta(2, 2.0), Workgroups::linear(1))
            .unwrap();
        // x * 2, then x * 3, queued back to back with no wait between.
        dev.scope(|s| {
            let (xs, n) = (s.bind(&mut x)?, s.bind_node(&mut scale)?);
            s.execute(n, &[Arg::Write(xs)])?;
            s.update_meta(n, &scale_meta(2, 3.0))?;
            s.execute(n, &[Arg::Write(xs)])
        })
        .unwrap();
        assert_eq!(
            x.download_vec::<F32>().unwrap(),
            vec![6.0, 12.0],
            "submit_every = {submit_every}"
        );
    }
}

#[test]
fn chained_runs_see_each_others_writes() {
    for submit_every in [1, 4, 0] {
        let Some(dev) = device(submit_every) else {
            return;
        };
        let mut x = dev.tensor_from(&F32::wrap(vec![1.0; 1000])).unwrap();
        let mut y = dev.tensor_from(&F32::wrap(vec![1.0; 1000])).unwrap();
        let mut add = dev
            .node(
                &shader(&ADD),
                &[MetaValue::Uint(1000)],
                Workgroups::linear(4),
            )
            .unwrap();
        // x += y, ten times, each reading the last one's result.
        dev.scope(|s| {
            let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
            for _ in 0..10 {
                s.execute(add, &[Arg::Write(x), Arg::Read(y)])?;
            }
            Ok::<_, crate::Error>(())
        })
        .unwrap();
        assert_eq!(
            x.download_vec::<F32>().unwrap(),
            vec![11.0; 1000],
            "submit_every = {submit_every}"
        );
    }
}

#[test]
fn ranges_copies_and_zero_round_trip() {
    let Some(dev) = device(1) else { return };
    let mut a = dev
        .tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap();
    a.upload_at(1, &F32::wrap(vec![20.0, 30.0])).unwrap();
    assert_eq!(
        F32::unwrap(a.download_range(0, 4).unwrap()),
        Some(vec![1.0, 20.0, 30.0, 4.0])
    );
    let mut b = dev.tensor_from(&F32::wrap(vec![0.0; 5])).unwrap();
    a.copy_into(1, &mut b, 2, 3).unwrap();
    assert_eq!(
        b.download_vec::<F32>().unwrap(),
        vec![0.0, 0.0, 20.0, 30.0, 4.0]
    );
    b.zero().unwrap();
    assert_eq!(b.download_vec::<F32>().unwrap(), vec![0.0; 5]);
}

#[test]
fn an_empty_tensor_round_trips() {
    let Some(dev) = device(1) else { return };
    let t = dev.alloc(DataKind::F32, 0).unwrap();
    assert_eq!(t.download_vec::<F32>().unwrap(), Vec::<f32>::new());
}

// SHADERS.md standard 10 for the WGSL code of every builtin.
#[test]
fn min_lens_cover_what_wgsl_code_touches() {
    let Some(dev) = device(1) else { return };
    crate::tools::builtins::tests::min_lens_cover_what_code_touches(&dev, shader);
}

// Nodes of one shader share its pipeline, each with its own meta; another
// shader gets its own.
#[test]
fn a_pipeline_is_made_once_per_shader() {
    let Some(dev) = device(1) else { return };
    let (add, scale) = (shader(&ADD), shader(&SCALE));
    let wg = Workgroups::linear(1);
    let mut first = dev.node(&add, &[MetaValue::Uint(1)], wg).unwrap();
    let mut both = dev.node(&add, &[MetaValue::Uint(2)], wg).unwrap();
    assert_eq!(dev.raw().pipelines.lock().unwrap().len(), 1);
    let _scale = dev.node(&scale, &scale_meta(1, 2.0), wg).unwrap();
    assert_eq!(dev.raw().pipelines.lock().unwrap().len(), 2);

    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 1.0])).unwrap();
    let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 10.0])).unwrap();
    dev.scope(|s| {
        let (x, y) = (s.bind(&mut x)?, s.bind(&mut y)?);
        let (first, both) = (s.bind_node(&mut first)?, s.bind_node(&mut both)?);
        s.execute(first, &[Arg::Write(x), Arg::Read(y)])?;
        s.execute(both, &[Arg::Write(x), Arg::Read(y)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![21.0, 11.0]);
}

// The safe layer adds no transfer or wait to a loop of runs on wgpu.
#[test]
fn a_steady_loop_moves_nothing_on_wgpu() {
    let Some(info) = WgpuBackend::choosable_devices().into_iter().next() else {
        eprintln!("no wgpu adapter; skipped");
        return;
    };
    crate::safe::trace::tests::a_steady_loop_moves_nothing(
        WgpuBackend::attach(info).unwrap(),
        &shader(&ADD),
    );
}
