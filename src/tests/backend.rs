use super::*;
use crate::backend::shader::{
    BindingRole, MetaField, MetaType, Native, NativeBinding, Shader, ShaderBinary, Signature,
    Workgroups,
};
use crate::backends::{CpuBackend, CpuBuffer};

fn cpu() -> CpuBackend {
    CpuBackend::attach(CpuBackend::choosable_devices().remove(0)).unwrap()
}

fn buffer_with(ctx: &CpuBackend, data: &[f32]) -> CpuBuffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let mut buf = ctx.alloc_raw(bytes.len() as u64).unwrap();
    // SAFETY: the buffer was made for exactly these bytes.
    unsafe { ctx.upload_raw(&mut buf, 0, bytes) };
    buf
}

fn read_back(ctx: &CpuBackend, buf: &CpuBuffer, len: usize) -> Vec<f32> {
    // SAFETY: every caller asks for no more than the buffer holds.
    let bytes = unsafe { ctx.download_raw(buf, 0, len as u64 * 4) };
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

static COPY_SHADER: Signature = Signature {
    name: "Copy",
    meta: &[],
    layout: &[
        BindingRole::Input(DataKind::F32),
        BindingRole::Output(DataKind::F32),
    ],
    // takes its length from the buffers, so it asks nothing of them
    min_lens: |_, _, l| l.fill(0),
    workgroup_size: [1, 1, 1],
};

fn copy_shader() -> Shader {
    Shader::new(&COPY_SHADER, ShaderBinary::Native(Native(copy)))
}

// x[0] = n as f32
fn meta_echo(meta: &[u32], bindings: &mut [NativeBinding]) {
    if let NativeBinding::Write(x) = &mut bindings[0] {
        x[..4].copy_from_slice(&(meta[0] as f32).to_ne_bytes());
    }
}

static META_SHADER: Signature = Signature {
    name: "MetaEcho",
    meta: &[MetaField {
        name: "n",
        ty: MetaType::Uint,
    }],
    layout: &[BindingRole::Output(DataKind::F32)],
    min_lens: |_, _, l| l.fill(1),
    workgroup_size: [1, 1, 1],
};

fn meta_shader() -> Shader {
    Shader::new(&META_SHADER, ShaderBinary::Native(Native(meta_echo)))
}

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
    let node = ctx
        .build_node(&copy_shader(), &[], Workgroups::linear(1))
        .unwrap();
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
    let node = ours
        .build_node(&copy_shader(), &[], Workgroups::linear(1))
        .unwrap();
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
    let mut node = ctx
        .build_node(&meta_shader(), &[1], Workgroups::linear(1))
        .unwrap();
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

// --- capture on the CPU backend ---

#[test]
fn capture_records_instead_of_running_and_replay_runs_in_order() {
    let ctx = cpu();
    let mut x = buffer_with(&ctx, &[0.0]);
    let mut y = buffer_with(&ctx, &[0.0]);
    let echo = ctx
        .build_node(&meta_shader(), &[5], Workgroups::linear(1))
        .unwrap();
    let copy = ctx
        .build_node(&copy_shader(), &[], Workgroups::linear(1))
        .unwrap();

    ctx.begin_capture();
    // SAFETY: slots follow each layout, every buffer holds one F32, and
    // nodes and buffers stay put and untouched past the replay's `synchronize`.
    unsafe {
        ctx.execute_raw(&echo, &mut [Access::Write(&mut x)])
            .unwrap();
        ctx.execute_raw(&copy, &mut [Access::Read(&x), Access::Write(&mut y)])
            .unwrap();
    }
    let capture = ctx.end_capture().unwrap();
    assert_eq!(
        read_back(&ctx, &y, 1),
        vec![0.0],
        "nothing ran while recording"
    );

    // SAFETY: as above.
    unsafe { ctx.replay(&capture).unwrap() };
    ctx.synchronize();
    assert_eq!(read_back(&ctx, &y, 1), vec![5.0], "echo ran before copy");
    ctx.release_capture(capture);
}

#[test]
fn replay_sees_update_meta() {
    let ctx = cpu();
    let mut x = buffer_with(&ctx, &[0.0]);
    let mut node = ctx
        .build_node(&meta_shader(), &[1], Workgroups::linear(1))
        .unwrap();

    ctx.begin_capture();
    // SAFETY: one Output slot, one F32; node and buffer stay put.
    unsafe {
        ctx.execute_raw(&node, &mut [Access::Write(&mut x)])
            .unwrap()
    };
    let capture = ctx.end_capture().unwrap();

    for n in [2, 3] {
        ctx.update_meta(&mut node, &[n]);
        // SAFETY: as above.
        unsafe { ctx.replay(&capture).unwrap() };
        ctx.synchronize();
        assert_eq!(read_back(&ctx, &x, 1), vec![n as f32]);
    }
}

#[test]
fn end_capture_without_begin_fails() {
    assert!(cpu().end_capture().is_err());
}

#[test]
#[should_panic(expected = "already recording")]
fn begin_capture_twice_panics() {
    let ctx = cpu();
    ctx.begin_capture();
    ctx.begin_capture();
}

#[test]
fn capture_rejects_another_devices_buffer() {
    let (ours, theirs) = (cpu(), cpu());
    let src = buffer_with(&theirs, &[1.0]);
    let mut dst = buffer_with(&ours, &[0.0]);
    let node = ours
        .build_node(&copy_shader(), &[], Workgroups::linear(1))
        .unwrap();
    ours.begin_capture();
    // SAFETY: the call must fail before recording anything.
    let result =
        unsafe { ours.execute_raw(&node, &mut [Access::Read(&src), Access::Write(&mut dst)]) };
    assert!(result.is_err());
    let capture = ours.end_capture().unwrap();
    // SAFETY: nothing was recorded, so the replay touches nothing.
    unsafe { ours.replay(&capture).unwrap() };
    assert_eq!(
        read_back(&ours, &dst, 1),
        vec![0.0],
        "the rejected call wasn't recorded"
    );
}

#[test]
fn replay_rejects_another_devices_capture() {
    let (ours, theirs) = (cpu(), cpu());
    theirs.begin_capture();
    let capture = theirs.end_capture().unwrap();
    // SAFETY: the call must fail before running anything.
    assert!(unsafe { ours.replay(&capture) }.is_err());
}

// --- areas on the CPU backend ---

#[test]
fn carved_buffers_sit_side_by_side_and_work_like_any_buffer() {
    let ctx = cpu();
    let area = ctx.reserve(24);
    let mut src = ctx.carve(&area, 12).unwrap();
    let mut dst = ctx.carve(&area, 12).unwrap();
    assert_eq!(dst.buffer().address() - src.buffer().address(), 12);

    // SAFETY: both parts hold 12 bytes.
    unsafe {
        ctx.upload_raw(
            src.buffer_mut(),
            0,
            bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]),
        )
    };
    let node = ctx
        .build_node(&copy_shader(), &[], Workgroups::linear(1))
        .unwrap();
    // SAFETY: COPY_SHADER's layout, both hold 3 F32s, untouched until `synchronize`.
    unsafe {
        ctx.execute_raw(
            &node,
            &mut [Access::Read(src.buffer()), Access::Write(dst.buffer_mut())],
        )
        .unwrap();
    }
    ctx.synchronize();
    assert_eq!(read_back(&ctx, dst.buffer(), 3), vec![1.0, 2.0, 3.0]);
    assert_eq!(read_back(&ctx, src.buffer(), 3), vec![1.0, 2.0, 3.0]);
}

#[test]
fn carve_rounds_to_words_and_fails_when_the_area_is_full() {
    let ctx = cpu();
    let area = ctx.reserve(10);
    assert_eq!(area.size_bytes(), 12);
    let zero = ctx.carve(&area, 0).unwrap();
    assert_eq!(zero.buffer().size_bytes(), 0);
    let five = ctx.carve(&area, 5).unwrap();
    assert_eq!(five.buffer().size_bytes(), 8);
    assert_eq!(area.remaining_bytes(), 4);
    assert!(ctx.carve(&area, 5).is_err());
    assert_eq!(area.remaining_bytes(), 4, "a failed carve takes nothing");
    ctx.carve(&area, 4).unwrap();
    assert_eq!(area.remaining_bytes(), 0);
}

#[test]
fn carve_rejects_another_devices_area() {
    let (ours, theirs) = (cpu(), cpu());
    let area = theirs.reserve(16);
    assert!(ours.carve(&area, 4).is_err());
}

// --- offsets, copies and clears ---

#[test]
fn upload_at_an_offset_and_download_a_range() {
    let ctx = cpu();
    let mut buf = buffer_with(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    // SAFETY: offsets and lengths are multiples of 4 and inside the 16 bytes.
    let middle = unsafe {
        ctx.upload_raw(&mut buf, 8, bytemuck::cast_slice(&[30.0f32]));
        ctx.download_raw(&buf, 4, 8)
    };
    assert_eq!(
        bytemuck::pod_collect_to_vec::<u8, f32>(&middle),
        vec![2.0, 30.0]
    );
    assert_eq!(read_back(&ctx, &buf, 4), vec![1.0, 2.0, 30.0, 4.0]);
}

#[test]
fn copy_raw_copies_a_range_between_buffers() {
    let ctx = cpu();
    let src = buffer_with(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let mut dst = buffer_with(&ctx, &[0.0; 4]);
    // SAFETY: both ranges are 8 bytes inside 16, at multiples of 4, and
    // nothing touches the buffers before `synchronize`.
    unsafe { ctx.copy_raw(&src, 4, &mut dst, 8, 8).unwrap() };
    ctx.synchronize();
    assert_eq!(read_back(&ctx, &dst, 4), vec![0.0, 0.0, 2.0, 3.0]);
}

#[test]
fn copy_raw_rejects_another_devices_buffer() {
    let (ours, theirs) = (cpu(), cpu());
    let src = buffer_with(&theirs, &[1.0]);
    let mut dst = buffer_with(&ours, &[0.0]);
    // SAFETY: the call must fail before copying anything.
    assert!(unsafe { ours.copy_raw(&src, 0, &mut dst, 0, 4) }.is_err());
    assert_eq!(read_back(&ours, &dst, 1), vec![0.0]);
}

#[test]
fn clear_raw_zeroes_the_whole_buffer() {
    let ctx = cpu();
    let mut buf = buffer_with(&ctx, &[1.0, 2.0, 3.0]);
    // SAFETY: nothing touches the buffer before `synchronize`.
    unsafe { ctx.clear_raw(&mut buf).unwrap() };
    ctx.synchronize();
    assert_eq!(read_back(&ctx, &buf, 3), vec![0.0; 3]);
}

#[test]
fn clear_raw_rejects_another_devices_buffer() {
    let (ours, theirs) = (cpu(), cpu());
    let mut buf = buffer_with(&theirs, &[1.0]);
    // SAFETY: the call must fail before clearing anything.
    assert!(unsafe { ours.clear_raw(&mut buf) }.is_err());
    assert_eq!(read_back(&theirs, &buf, 1), vec![1.0]);
}

static WIDE_SHADER: Signature = Signature {
    name: "Wide",
    meta: &[],
    layout: &[],
    min_lens: |_, _, _| {},
    workgroup_size: [256, 1, 1],
};

#[test]
fn cover_rounds_up_to_whole_workgroups() {
    let wide = Shader::new(&WIDE_SHADER, ShaderBinary::Native(Native(copy)));
    let x = |len| Workgroups::cover(len, &wide).dims();
    assert_eq!(x(0), [0, 1, 1]);
    assert_eq!(x(1), [1, 1, 1]);
    assert_eq!(x(256), [1, 1, 1]);
    assert_eq!(x(257), [2, 1, 1]);
    assert_eq!(x(1000), [4, 1, 1]);
    assert_eq!(Workgroups::cover(5, &copy_shader()).dims(), [5, 1, 1]);
}
