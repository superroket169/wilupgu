// partial[w] = sum of x[i] over workgroup w's slice   (i < n; native runs one workgroup)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::backend::shader::CpuBinding;

// Unlike the wgsl version, the CPU reference needs no multi-pass reduction:
// one thread, one loop, done.
pub(crate) fn entry(bindings: &[CpuBinding]) {
    let n = read_u32(find(bindings, 0))[0] as usize;
    let x = read_f32(find(bindings, 1));
    let mut partial = read_f32(find(bindings, 2));
    partial[0] = x[..n].iter().sum();
    write_f32(find(bindings, 2), &partial);
}
