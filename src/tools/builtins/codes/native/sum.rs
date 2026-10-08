// partial[w] = sum of x[i] over workgroup w's slice   (i < n; native runs one workgroup)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

// Unlike the wgsl version, the CPU reference needs no multi-pass reduction:
// one thread, one loop, done.
pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let n = meta[0] as usize;
    let x = read_f32(bindings, 1);
    let mut partial = read_f32(bindings, 2);
    partial[0] = x[..n].iter().sum();
    write_f32(bindings, 2, &partial);
}
