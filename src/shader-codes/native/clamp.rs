use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn clamp(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let lo = f32::from_bits(meta[0]);
    let hi = f32::from_bits(meta[1]);
    for xi in x.iter_mut() {
        *xi = xi.clamp(lo, hi);
    }
    write_f32(find(bindings, 0), &x);
}
