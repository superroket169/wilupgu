// C[n] += sum_k A[k] * B[k * N + n]   (M = 1; A: 1xK, B: KxN, C: 1xN)
use super::common::{read_f32, write_f32};
use crate::backend::shader::NativeBinding;

// M is ignored, like the WGSL version: one row.
pub(crate) fn entry(meta: &[u32], bindings: &mut [NativeBinding]) {
    let (n, k) = (meta[1] as usize, meta[2] as usize);
    let a = read_f32(bindings, 1);
    let b = read_f32(bindings, 2);
    let mut c = read_f32(bindings, 3);
    for col in 0..n {
        let mut sum = 0.0f32;
        for kk in 0..k {
            sum += a[kk] * b[kk * n + col];
        }
        c[col] += sum;
    }
    write_f32(bindings, 3, &c);
}
