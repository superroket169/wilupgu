// C[m * N + n] = sum_k A[m * K + k] * B[k * N + n]   (A: MxK, B: KxN, C: MxN)
use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn entry(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 0));
    let (m, n, k) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let a = read_f32(find(bindings, 1));
    let b = read_f32(find(bindings, 2));
    let mut c = read_f32(find(bindings, 3));
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    write_f32(find(bindings, 3), &c);
}
