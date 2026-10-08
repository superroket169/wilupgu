// x[i] = value   (i < n)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let (n, value) = (meta[0] as usize, f32::from_bits(meta[1]));
    let mut x = read_f32(bindings, 1);
    for xi in x.iter_mut().take(n) {
        *xi = value;
    }
    write_f32(bindings, 1, &x);
}
