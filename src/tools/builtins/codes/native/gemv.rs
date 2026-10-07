// C[n] = sum_k A[k] * B[k * N + n]   (M = 1; A: 1xK, B: KxN, C: 1xN)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::backend::shader::CpuBinding;

// M is ignored, like the WGSL version: one row.
pub(crate) fn entry(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 0));
    let (n, k) = (meta[1] as usize, meta[2] as usize);
    let a = read_f32(find(bindings, 1));
    let b = read_f32(find(bindings, 2));
    let mut c = read_f32(find(bindings, 3));
    for col in 0..n {
        let mut sum = 0.0f32;
        for kk in 0..k {
            sum += a[kk] * b[kk * n + col];
        }
        c[col] = sum;
    }
    write_f32(find(bindings, 3), &c);
}
