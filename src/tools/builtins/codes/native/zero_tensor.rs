// x[i] = 0   (i < n)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let n = meta[0] as usize;
    let mut x = read_f32(bindings, 1);
    for xi in x.iter_mut().take(n) {
        *xi = 0.0;
    }
    write_f32(bindings, 1, &x);
}
