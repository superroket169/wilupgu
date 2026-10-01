use crate::core::shader::CpuBinding;

fn find(bindings: &[CpuBinding], slot: u32) -> &CpuBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn read_f32(b: &CpuBinding) -> Vec<f32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, f32>(&g)
}

fn read_u32(b: &CpuBinding) -> Vec<u32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, u32>(&g)
}

fn write_f32(b: &CpuBinding, data: &[f32]) {
    let mut g = b.buffer.lock().unwrap();
    g.copy_from_slice(bytemuck::cast_slice(data));
}

// Unlike the wgsl version, the CPU reference needs no multi-pass reduction:
// one thread, one loop, done.
pub(crate) fn sum(bindings: &[CpuBinding]) {
    let x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 2));
    let n = meta[0] as usize;
    let total: f32 = x[..n].iter().sum();
    write_f32(find(bindings, 1), &[total]);
}
