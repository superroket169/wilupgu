// partial[w] = sum of a[i] * b[i] over workgroup w's slice   (i < n; native runs one workgroup)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::backend::shader::CpuBinding;

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let n = read_u32(find(bindings, 0))[0] as usize;
    let a = read_f32(find(bindings, 1));
    let b = read_f32(find(bindings, 2));
    let mut partial = read_f32(find(bindings, 3));
    partial[0] = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    write_f32(find(bindings, 3), &partial);
}
