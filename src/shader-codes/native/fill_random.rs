// x[i] = hash(i ^ seed) / 0xFFFFFFFF   (i < n)
use super::common::{find, read_f32, read_u32, write_f32};
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

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 0));
    let (n, seed) = (meta[0] as usize, meta[1]);
    let mut x = read_f32(find(bindings, 1));
    for (i, xi) in x.iter_mut().enumerate().take(n) {
        *xi = hash(i as u32 ^ seed) as f32 / u32::MAX as f32;
    }
    write_f32(find(bindings, 1), &x);
}
