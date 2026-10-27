use super::*;
use crate::backend::dtype::{DataType, F32};
use crate::backend::shader::{Ptx, Shader, ShaderBinary};
use crate::safe::{Arg, ComputeDevice, DeviceConfig, MetaValue};
use crate::tools::builtins::{Builtin, ADD, ALL, SCALE};
use crate::tools::compile::compilers::nvrtc::Nvrtc;
use crate::tools::compile::ShaderCompiler;

// The first CUDA device, or `None`, with a note, without one or without NVRTC.
fn device() -> Option<ComputeDevice<CudaBackend>> {
    let Some(info) = CudaBackend::choosable_devices().into_iter().next() else {
        eprintln!("no CUDA device; skipped");
        return None;
    };
    // SAFETY: it only tries to open the library.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        eprintln!("NVRTC isn't installed; skipped");
        return None;
    }
    let cuda = CudaBackend::attach(info).unwrap();
    Some(ComputeDevice::new(cuda, DeviceConfig::default()))
}

fn shader(builtin: &'static Builtin) -> Shader {
    let ptx: Ptx = Nvrtc::default().compile(&builtin.cuda).unwrap();
    Shader::new(&builtin.signature, ShaderBinary::Ptx(ptx))
}

fn scale_meta(n: u32, factor: f32) -> [MetaValue; 2] {
    [MetaValue::Uint(n), MetaValue::Float(factor)]
}

#[test]
fn choosable_devices_never_panics() {
    let _ = CudaBackend::choosable_devices();
}

#[test]
fn every_builtin_loads() {
    let Some(dev) = device() else { return };
    for b in ALL {
        let meta: Vec<MetaValue> = b
            .signature
            .meta
            .iter()
            .map(|field| match field.ty {
                crate::backend::shader::MetaType::Uint => MetaValue::Uint(1),
                crate::backend::shader::MetaType::Float => MetaValue::Float(0.5),
            })
            .collect();
        dev.node(&shader(b), &meta, Workgroups::linear(1))
            .unwrap_or_else(|e| panic!("`{}`: {e}", b.signature.name));
    }
}

#[test]
fn add_runs() {
    let Some(dev) = device() else { return };
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0])).unwrap();
    let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 20.0, 30.0])).unwrap();
    let mut add = dev
        .node(&shader(&ADD), &[MetaValue::Uint(3)], Workgroups::linear(1))
        .unwrap();
    dev.scope(|s| {
        let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
        s.run(add, &[Arg::Write(x), Arg::Read(y)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn update_meta_reaches_only_the_runs_after_it() {
    let Some(dev) = device() else { return };
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
    let mut scale = dev
        .node(&shader(&SCALE), &scale_meta(2, 2.0), Workgroups::linear(1))
        .unwrap();
    // x * 2, then x * 3, queued back to back with no wait between.
    dev.scope(|s| {
        let (xs, n) = (s.bind(&mut x)?, s.bind_node(&mut scale)?);
        s.run(n, &[Arg::Write(xs)])?;
        s.update_meta(n, &scale_meta(2, 3.0))?;
        s.run(n, &[Arg::Write(xs)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![6.0, 12.0]);
}

#[test]
fn replay_runs_with_the_current_meta() {
    let Some(dev) = device() else { return };
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
    let mut scale = dev
        .node(&shader(&SCALE), &scale_meta(2, 2.0), Workgroups::linear(1))
        .unwrap();
    let recorded = dev.scope(|s| {
        let (xs, n) = (s.bind(&mut x)?, s.bind_node(&mut scale)?);
        s.capture(|| s.run(n, &[Arg::Write(xs)]))
    });
    let recording = match recorded {
        Ok(recording) => recording,
        Err(e) if e.to_string().contains("can't capture") => {
            eprintln!("{e}; skipped");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    // Recording ran nothing.
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![1.0, 2.0]);
    dev.scope(|s| {
        let (_, n) = (s.bind(&mut x)?, s.bind_node(&mut scale)?);
        s.replay(&recording)?;
        s.update_meta(n, &scale_meta(2, 3.0))?;
        s.replay(&recording)
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![6.0, 12.0]);
}

#[test]
fn ranges_copies_and_zero_round_trip() {
    let Some(dev) = device() else { return };
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
fn area_buffers_run_like_allocated_ones() {
    let Some(dev) = device() else { return };
    let area = dev.reserve(1024);
    let mut x = area.alloc(crate::backend::dtype::DataKind::F32, 3).unwrap();
    let mut y = area.alloc(crate::backend::dtype::DataKind::F32, 3).unwrap();
    x.upload(&F32::wrap(vec![1.0, 2.0, 3.0])).unwrap();
    y.upload(&F32::wrap(vec![1.0, 1.0, 1.0])).unwrap();
    let mut add = dev
        .node(&shader(&ADD), &[MetaValue::Uint(3)], Workgroups::linear(1))
        .unwrap();
    dev.scope(|s| {
        let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
        s.run(add, &[Arg::Write(x), Arg::Read(y)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>().unwrap(), vec![2.0, 3.0, 4.0]);
}
