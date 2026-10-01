use crate::core::shader::CpuBinding;

fn find(bindings: &[CpuBinding], slot: u32) -> &CpuBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn read_u32(b: &CpuBinding) -> Vec<u32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, u32>(&g)
}

fn write_f32(b: &CpuBinding, data: &[f32]) {
    let mut g = b.buffer.lock().unwrap();
    g.copy_from_slice(bytemuck::cast_slice(data));
}

pub(crate) fn fill_constant(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 1));
    let len = meta[0] as usize;
    let value = f32::from_bits(meta[1]);
    write_f32(find(bindings, 0), &vec![value; len]);
}
