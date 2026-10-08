// dst[c * rows + r] = src[r * cols + c]   (r < rows, c < cols)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let (rows, cols) = (meta[0] as usize, meta[1] as usize);
    let src = read_f32(bindings, 1);
    let mut dst = read_f32(bindings, 2);
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    write_f32(bindings, 2, &dst);
}
