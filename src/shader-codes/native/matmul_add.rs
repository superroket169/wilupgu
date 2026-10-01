use crate::core::shader::CpuBinding;

fn find(bindings: &[CpuBinding], slot: u32) -> &CpuBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn read_f32(b: &CpuBinding) -> Vec<f32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, f32>(&g)
}

fn read_u32(b: &CpuBinding) -> Vec<u32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, u32>(&g)
}

fn write_f32(b: &CpuBinding, data: &[f32]) {
    let mut g = b.buffer.lock().unwrap();
    g.copy_from_slice(bytemuck::cast_slice(data));
}

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
