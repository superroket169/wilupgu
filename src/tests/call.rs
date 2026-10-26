use super::*;
use crate::backend::dtype::DataKind;
use crate::backend::shader::{
    BindingRole, MetaField, MetaType, Native, NativeBinding, Shader, ShaderBinary, Signature,
    Workgroups,
};
use crate::backends::{CpuBackend, CpuInfo};
use crate::safe::DeviceConfig;
use crate::tools::plan::deferred::Resolvable;
use crate::tools::plan::node::{MetaSource, NodeDesc};
use crate::tools::plan::tensor::{TensorDesc, TensorInit, TensorSpec};

fn copy(meta: &[u32], bindings: &mut [NativeBinding]) {
    let [NativeBinding::Write(dst), NativeBinding::Read(src)] = bindings else {
        panic!("copy_like takes a written and a read slot")
    };
    let n = meta[0] as usize * 4;
    dst[..n].copy_from_slice(&src[..n]);
}

static COPY_LIKE: Signature = Signature {
    name: "copy_like",
    meta: &[MetaField {
        name: "n",
        ty: MetaType::Uint,
    }],
    layout: &[
        BindingRole::Output(DataKind::F32),
        BindingRole::Input(DataKind::F32),
    ],
    min_lens: |m, _, l| l.fill(m[0] as usize),
    workgroup_size: [1, 1, 1],
};

fn copy_like() -> Shader {
    Shader::new(&COPY_LIKE, ShaderBinary::Native(Native(copy)))
}

fn tensor_id() -> GlobalId<TensorSpec> {
    TensorSpec::new(TensorDesc {
        kind: DataKind::F32,
        len: Resolvable::fixed(1),
        init: TensorInit::Blank,
    })
    .unwrap()
    .id()
}

fn node(n: Resolvable<u32>) -> NodeSpec {
    NodeSpec::new(NodeDesc {
        shader: copy_like(),
        meta: vec![MetaSource::Uint(n)],
        workgroups: Resolvable::fixed(Workgroups { x: 1, y: 1, z: 1 }),
    })
    .unwrap()
}

#[test]
fn new_wants_one_tensor_per_slot() {
    let (a, b) = (tensor_id(), tensor_id());
    let call = |tensors| {
        CallSpec::new(CallDesc {
            node: node(Resolvable::fixed(1)),
            tensors,
        })
    };
    assert!(call(vec![a]).is_err());
    assert!(call(vec![a, b, a]).is_err());
    assert!(call(vec![a, b]).is_ok());
    assert!(call(vec![a, a]).is_ok(), "two slots can name one tensor");
}

#[test]
fn ready_waits_for_the_node() {
    let n = Resolvable::new();
    let resolver = n.resolver().unwrap();
    let call = CallSpec::new(CallDesc {
        node: node(n),
        tensors: vec![tensor_id(), tensor_id()],
    })
    .unwrap();
    assert!(call.ready().is_err());
    resolver.resolve(1);
    assert!(call.ready().is_ok());
}

#[test]
fn build_gives_the_node_and_names_the_tensors() {
    let (a, b) = (tensor_id(), tensor_id());
    let spec = CallSpec::new(CallDesc {
        node: node(Resolvable::fixed(3)),
        tensors: vec![a, b],
    })
    .unwrap();
    let dev = ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    );
    let call = spec.build(&dev).unwrap();
    assert_eq!(call.node().signature().name, "copy_like");
    assert_eq!(call.tensors(), &[a, b]);
    assert_ne!(
        spec.id(),
        CallSpec::new(CallDesc {
            node: node(Resolvable::fixed(1)),
            tensors: vec![a, b]
        })
        .unwrap()
        .id()
    );
}
