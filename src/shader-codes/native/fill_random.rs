use super::common::{find, read_u32, write_f32};
use crate::core::shader::CpuBinding;

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
