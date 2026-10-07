// dst[c * rows + r] = src[r * cols + c]   (r < rows, c < cols)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::backend::shader::CpuBinding;

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 0));
    let (rows, cols) = (meta[0] as usize, meta[1] as usize);
    let src = read_f32(find(bindings, 1));
    let mut dst = read_f32(find(bindings, 2));
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    write_f32(find(bindings, 2), &dst);
}
