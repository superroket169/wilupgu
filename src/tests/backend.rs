use super::*;
use crate::backend::shader::{
    BindingRole, MetaField, MetaType, NativeBinding, NativeCode, ShaderCode,
};
use crate::backends::{CpuBackend, CpuBuffer};

fn cpu() -> CpuBackend {
    CpuBackend::attach(CpuBackend::choosable_devices().remove(0)).unwrap()
}

fn buffer_with(ctx: &CpuBackend, data: &[f32]) -> CpuBuffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let mut buf = ctx.alloc_raw(bytes.len() as u64).unwrap();
    // SAFETY: the buffer was made for exactly these bytes.
    unsafe { ctx.upload_raw(&mut buf, bytes) };
    buf
}

fn read_back(ctx: &CpuBackend, buf: &CpuBuffer, len: usize) -> Vec<f32> {
    // SAFETY: every caller asks for no more than the buffer holds.
    let bytes = unsafe { ctx.download_raw(buf, len as u64 * 4) };
    bytemuck::pod_collect_to_vec(&bytes)
}

// dst = src, element by element
fn copy(_meta: &[u32], bindings: &mut [NativeBinding]) {
    let src = match &bindings[0] {
        NativeBinding::Read(b) => b.to_vec(),
        NativeBinding::Write(_) => unreachable!(),
    };
    if let NativeBinding::Write(dst) = &mut bindings[1] {
        dst.copy_from_slice(&src);
    }
}

static COPY_SHADER: Shader = Shader {
    name: "Copy",
    meta: &[],
    layout: &[
        BindingRole::Input(DataKind::F32),
        BindingRole::Output(DataKind::F32),
    ],
    workgroup_size: [1, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(copy)),
        ..ShaderCode::NONE
    },
};

// x[0] = n as f32
fn meta_echo(meta: &[u32], bindings: &mut [NativeBinding]) {
    if let NativeBinding::Write(x) = &mut bindings[0] {
        x[..4].copy_from_slice(&(meta[0] as f32).to_ne_bytes());
    }
}

static META_SHADER: Shader = Shader {
    name: "MetaEcho",
    meta: &[MetaField {
        name: "n",
        ty: MetaType::Uint,
    }],
    layout: &[BindingRole::Output(DataKind::F32)],
    workgroup_size: [1, 1, 1],
    shader_code: ShaderCode {
        native: Some(NativeCode::new(meta_echo)),
        ..ShaderCode::NONE
    },
};

#[test]
fn topology_enumerate_then_attach() {
    let devices = CpuBackend::choosable_devices();
    assert_eq!(devices.len(), 1);
    let ctx = CpuBackend::attach(devices[0].clone()).unwrap();
    assert_eq!(CpuBackend::NAME, "cpu");
    assert_eq!(ctx.info().label(), "cpu");
}

#[test]
fn alloc_rounds_up_to_whole_words() {
    let buf = cpu().alloc_raw(5).unwrap();
    assert_eq!(buf.size_bytes(), 8);
}

#[test]
fn upload_then_download_round_trips() {
    let ctx = cpu();
    let buf = buffer_with(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_back(&ctx, &buf, 4), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn download_returns_only_the_asked_bytes() {
    let ctx = cpu();
    let buf = buffer_with(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_back(&ctx, &buf, 2), vec![1.0, 2.0]);
}

#[test]
fn address_survives_a_move() {
    let buf = cpu().alloc_raw(16).unwrap();
    let before = buf.address();
    let moved = buf;
    assert_eq!(moved.address(), before);
}

#[test]
fn execute_runs_the_node_on_its_bindings() {
    let ctx = cpu();
    let src = buffer_with(&ctx, &[1.0, 2.0, 3.0]);
    let mut dst = buffer_with(&ctx, &[0.0; 3]);
    let node = ctx.build_node(&COPY_SHADER, &[], Workgroups::linear(1));
    // SAFETY: slot order and roles follow COPY_SHADER's layout, both buffers
    // hold 3 F32s, and nothing touches them before `synchronize`.
    unsafe {
        ctx.execute_raw(&node, &mut [Access::Read(&src), Access::Write(&mut dst)])
            .unwrap();
    }
    ctx.synchronize();
    assert_eq!(read_back(&ctx, &dst, 3), vec![1.0, 2.0, 3.0]);
}

#[test]
fn execute_rejects_another_devices_buffer() {
    let (ours, theirs) = (cpu(), cpu());
    let src = buffer_with(&theirs, &[1.0]);
    let mut dst = buffer_with(&ours, &[0.0]);
    let node = ours.build_node(&COPY_SHADER, &[], Workgroups::linear(1));
    // SAFETY: as above; the call must fail before running anything.
    let err =
        unsafe { ours.execute_raw(&node, &mut [Access::Read(&src), Access::Write(&mut dst)]) }
            .unwrap_err();
    assert!(err.contains("another device"), "{err}");
}

#[test]
fn update_meta_reaches_the_next_run() {
    let ctx = cpu();
    let mut x = buffer_with(&ctx, &[0.0]);
    let mut node = ctx.build_node(&META_SHADER, &[1], Workgroups::linear(1));
    ctx.update_meta(&mut node, &[7]);
    // SAFETY: one Output slot, one F32, untouched until `synchronize`.
    unsafe {
        ctx.execute_raw(&node, &mut [Access::Write(&mut x)])
            .unwrap()
    };
    ctx.synchronize();
    assert_eq!(read_back(&ctx, &x, 1), vec![7.0]);
}

#[test]
fn supports_p2p_defaults_to_false() {
    let (a, b) = (cpu(), cpu());
    assert!(!a.supports_p2p(&b));
}
