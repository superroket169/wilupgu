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

fn write_f32(b: &CpuBinding, data: &[f32]) {
    let mut g = b.buffer.lock().unwrap();
    g.copy_from_slice(bytemuck::cast_slice(data));
}

pub(crate) fn max(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let y = read_f32(find(bindings, 1));
    for (xi, yi) in x.iter_mut().zip(y.iter()) {
        *xi = xi.max(*yi);
    }
    write_f32(find(bindings, 0), &x);
}
