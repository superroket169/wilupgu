// partial[w] = sum of a[i] * b[i] over workgroup w's slice   (i < n; native runs one workgroup)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let n = meta[0] as usize;
    let a = read_f32(bindings, 1);
    let b = read_f32(bindings, 2);
    let mut partial = read_f32(bindings, 3);
    partial[0] = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    write_f32(bindings, 3, &partial);
}
