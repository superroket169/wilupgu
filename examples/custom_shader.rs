use std::borrow::Cow;

use wilupgu::backend::shader::{
    BindingRole, MetaField, MetaType, Native, NativeBinding, Signature, Wgsl,
};
use wilupgu::backend::Dispatch;
use wilupgu::backends::CpuBackend;
use wilupgu::prelude::*;

// y[i] = a * x[i] + y[i] for i < n. The signature is what the safe layer
// checks every run against: the meta's fields, each slot's role and kind,
// and how many elements each slot needs.
static SAXPY: Signature = Signature {
    name: "saxpy",
    meta: &[
        MetaField {
            name: "n",
            ty: MetaType::Uint,
        },
        MetaField {
            name: "a",
            ty: MetaType::Float,
        },
    ],
    layout: &[
        BindingRole::Accumulate(DataKind::F32),
        BindingRole::Input(DataKind::F32),
    ],
    // Both slots need n elements; a shorter tensor is rejected before the
    // run reaches the device, so the code below never checks lengths.
    min_lens: |meta, _, lens| lens.fill(meta[0] as usize),
    workgroup_size: [256, 1, 1],
};

// The meta is binding 0; the slots follow in the layout's order.
const WGSL: &str = r#"
struct ShaderMeta {
    n: u32,
    a: f32,
}

@group(0) @binding(0) var<storage, read> shader_meta: ShaderMeta;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn entry(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;
    if (idx >= shader_meta.n) {
        return;
    }
    y[idx] = shader_meta.a * x[idx] + y[idx];
}
"#;

// The same shader as native code, for the CPU backend. The meta comes as
// words; a float field is its bits.
fn native(meta: &[u32], bindings: &mut [NativeBinding]) {
    let [NativeBinding::Write(y), NativeBinding::Read(x)] = bindings else {
        panic!("saxpy takes a written and a read slot")
    };
    let (n, a) = (meta[0] as usize, f32::from_bits(meta[1]));
    let y = bytemuck::cast_slice_mut::<u8, f32>(y);
    let x = bytemuck::cast_slice::<u8, f32>(x);
    for i in 0..n {
        y[i] += a * x[i];
    }
}

// The first of the two forms the backend takes.
fn saxpy<B: Dispatch>() -> Result<Shader, Error> {
    [
        ShaderBinary::Native(Native(native)),
        ShaderBinary::Wgsl(Wgsl(Cow::Borrowed(WGSL))),
    ]
    .into_iter()
    .find(B::accepts)
    .map(|binary| Shader::new(&SAXPY, binary))
    .ok_or(Error::UnsupportedBinary { shader: SAXPY.name })
}

fn run<B: Backend + Topology>(x: &[f32], y: &[f32], n: u32, a: f32) -> Result<Vec<f32>, Error> {
    let dev = ComputeDevice::<B>::first()?;
    let mut x = dev.tensor_from(&F32::wrap(x.to_vec()))?;
    let mut y = dev.tensor_from(&F32::wrap(y.to_vec()))?;
    let shader = saxpy::<B>()?;
    let wg = Workgroups::cover(n, &shader);
    let mut node = dev.node(&shader, &[MetaValue::Uint(n), MetaValue::Float(a)], wg)?;
    dev.scope(|s| {
        let (y, x, node) = (s.bind(&mut y)?, s.bind(&mut x)?, s.bind_node(&mut node)?);
        s.execute(node, &[Arg::Write(y), Arg::Read(x)])
    })?;
    y.download_vec::<F32>()
}

fn main() -> Result<(), Error> {
    let (x, y) = ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);
    assert_eq!(run::<CpuBackend>(&x, &y, 3, 2.0)?, vec![12.0, 24.0, 36.0]);

    // n = 4 asks for 4 elements of tensors that hold 3: the run fails its
    // checks and never reaches the device.
    let err = run::<CpuBackend>(&x, &y, 4, 2.0).unwrap_err();
    println!("n = 4 on 3 elements: {err}");

    #[cfg(feature = "wgpu")]
    println!(
        "wgpu: {:?}",
        run::<wilupgu::backends::WgpuBackend>(&x, &y, 3, 2.0)?
    );
    Ok(())
}
