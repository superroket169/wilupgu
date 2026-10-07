// x[i] = value   (i < n)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::backend::shader::CpuBinding;

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 0));
    let (n, value) = (meta[0] as usize, f32::from_bits(meta[1]));
    let mut x = read_f32(find(bindings, 1));
    for xi in x.iter_mut().take(n) {
        *xi = value;
    }
    write_f32(find(bindings, 1), &x);
}
