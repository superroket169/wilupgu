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

// Must match wgsl/fill_random.wgsl's `hash` exactly -- that's what makes the
// two backends' output comparable at all.
fn hash(v: u32) -> u32 {
    let mut h = v;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    h
}

pub(crate) fn fill_random(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 1));
    let (len, seed) = (meta[0], meta[1]);
    let data: Vec<f32> = (0..len)
        .map(|i| hash(i ^ seed) as f32 / u32::MAX as f32)
        .collect();
    write_f32(find(bindings, 0), &data);
}
