// x[i] = 0   (i < n)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 0));
    let n = meta[0] as usize;
    let mut x = read_f32(find(bindings, 1));
    for xi in x.iter_mut().take(n) {
        *xi = 0.0;
    }
    write_f32(find(bindings, 1), &x);
}
