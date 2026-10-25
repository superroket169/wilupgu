use super::*;
use crate::backend::dtype::{DataKind, DataType, F32};
use crate::backend::shader::{BindingRole, MetaField, NativeBinding, NativeCode, ShaderCode};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::{Arg, DeviceConfig};
use crate::tools::plan::deferred::Relation;

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

fn one_wg() -> Resolvable<Workgroups> {
    Resolvable::fixed(Workgroups { x: 1, y: 1, z: 1 })
}

fn scale_desc(n: Resolvable<u32>, factor: MetaSource) -> NodeDesc {
    NodeDesc {
        shader: &SCALE_LIKE,
        meta: vec![MetaSource::Uint(n), factor],
        workgroups: one_wg(),
    }
}

#[test]
fn new_checks_the_meta_against_the_shader() {
    let n = || Resolvable::fixed(4);
    let mut short = scale_desc(n(), MetaSource::Float(Resolvable::fixed(2.0)));
    short.meta.pop();
    assert!(NodeSpec::new(short).is_err());

    let wrong_type = scale_desc(n(), MetaSource::Uint(Resolvable::fixed(2)));
    assert!(NodeSpec::new(wrong_type).is_err());

    let per_run = scale_desc(n(), MetaSource::FloatPerRun(Dynamic::new()));
    assert!(NodeSpec::new(per_run).is_ok());
}

#[test]
fn ready_waits_for_once_values_and_workgroups_only() {
    let n = Resolvable::new();
    let n_resolver = n.resolver().unwrap();
    let wg = Resolvable::new();
    let wg_resolver = wg.resolver().unwrap();
    let mut desc = scale_desc(n, MetaSource::FloatPerRun(Dynamic::new()));
    desc.workgroups = wg;
    let spec = NodeSpec::new(desc).unwrap();

    assert!(spec.ready().is_err());
    n_resolver.resolve(4);
    assert!(spec.ready().is_err());
    wg_resolver.resolve(Workgroups { x: 1, y: 1, z: 1 });
    // the per-run value was never set, and isn't needed to build
    assert!(spec.ready().is_ok());
}

#[test]
fn build_uses_the_resolved_meta_and_workgroups() {
    let n = Resolvable::new();
    let workgroups = Relation::new([&n], |[n]: [u32; 1]| Workgroups {
        x: n.div_ceil(256),
        y: 1,
        z: 1,
    });
    let mut desc = scale_desc(n.clone(), MetaSource::Float(Resolvable::fixed(3.0)));
    desc.workgroups = workgroups;
    let spec = NodeSpec::new(desc).unwrap();
    n.resolver().unwrap().resolve(4);

    let dev = ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    );
    let mut node = spec.build(&dev).unwrap();
    assert_eq!(node.workgroups(), Workgroups { x: 1, y: 1, z: 1 });
    let mut x = dev
        .tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    dev.scope(|s| {
        let (node, x) = (s.bind_node(&mut node)?, s.bind(&mut x)?);
        s.run(node, &[Arg::Write(x)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>(), Ok(vec![3.0, 6.0, 9.0, 12.0]));
}

#[test]
fn build_writes_zero_for_a_per_run_value() {
    let spec = NodeSpec::new(scale_desc(
        Resolvable::fixed(2),
        MetaSource::FloatPerRun(Dynamic::new()),
    ))
    .unwrap();
    let dev = ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    );
    let mut node = spec.build(&dev).unwrap();
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0])).unwrap();
    dev.scope(|s| {
        let (node, x) = (s.bind_node(&mut node)?, s.bind(&mut x)?);
        s.run(node, &[Arg::Write(x)])
    })
    .unwrap();
    assert_eq!(x.download_vec::<F32>(), Ok(vec![0.0, 0.0]));
}
