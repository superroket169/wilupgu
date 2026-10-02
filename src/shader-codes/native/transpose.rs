use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn transpose(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 2));
    let (rows, cols) = (meta[0] as usize, meta[1] as usize);
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    write_f32(find(bindings, 1), &dst);
}
