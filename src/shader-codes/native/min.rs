// x[i] = min(x[i], y[i])   (i < n)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let n = read_u32(find(bindings, 0))[0] as usize;
    let mut x = read_f32(find(bindings, 1));
    let y = read_f32(find(bindings, 2));
    for (xi, yi) in x.iter_mut().zip(&y).take(n) {
        *xi = xi.min(*yi);
    }
    write_f32(find(bindings, 1), &x);
}
