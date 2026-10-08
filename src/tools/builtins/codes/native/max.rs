// x[i] = max(x[i], y[i])   (i < n)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let n = meta[0] as usize;
    let mut x = read_f32(bindings, 1);
    let y = read_f32(bindings, 2);
    for (xi, yi) in x.iter_mut().zip(&y).take(n) {
        *xi = xi.max(*yi);
    }
    write_f32(bindings, 1, &x);
}
