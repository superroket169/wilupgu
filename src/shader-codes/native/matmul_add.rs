use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

// C[row, col] += sum_k A[row, k] * B[k, col]   (accumulating matmul)
pub(crate) fn matmul_add(bindings: &[CpuBinding]) {
    let a = read_f32(find(bindings, 0));
    let b = read_f32(find(bindings, 1));
    let mut c = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 3));
    let (m, n, k) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] += sum;
        }
    }
    write_f32(find(bindings, 2), &c);
}
