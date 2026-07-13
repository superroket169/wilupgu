use crate::shader::CpuBinding;

// ========================================
//            Binding helpers
// ========================================

fn find(bindings: &[CpuBinding], slot: u32) -> &CpuBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn read_f32(b: &CpuBinding) -> Vec<f32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::cast_slice::<u8, f32>(&g).to_vec()
}

fn read_u32(b: &CpuBinding) -> Vec<u32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::cast_slice::<u8, u32>(&g).to_vec()
}

fn write_f32(b: &CpuBinding, data: &[f32]) {
    let mut g = b.buffer.lock().unwrap();
    g.copy_from_slice(bytemuck::cast_slice(data));
}

// ========================================
//          Kernel implementations
// ========================================

// C[row, col] = sum_k A[row, k] * B[k, col]   (A: MxK, B: KxN, C: MxN)
pub(crate) fn matmul(bindings: &[CpuBinding]) {
    let a = read_f32(find(bindings, 0));
    let b = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (m, n, k) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    let mut c = read_f32(find(bindings, 2));
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    write_f32(find(bindings, 2), &c);
}

// C[row, col] = sum_k A[row, k] * B[col, k]   (A: MxK, B: NxK, C: MxN) -- B read transposed
pub(crate) fn matmul_trp(bindings: &[CpuBinding]) {
    let a = read_f32(find(bindings, 0));
    let b = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (m, n, k) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    let mut c = read_f32(find(bindings, 2));
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[row * k + kk] * b[col * k + kk];
            }
            c[row * n + col] = sum;
        }
    }
    write_f32(find(bindings, 2), &c);
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

pub(crate) fn causal_mask(bindings: &[CpuBinding]) {
    let mut scores = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let seq_len = meta[0] as usize;
    let scale = f32::from_bits(meta[1]);

    for row in 0..seq_len {
        for col in 0..seq_len {
            let idx = row * seq_len + col;
            scores[idx] = if col > row {
                -1_000_000_000.0
            } else {
                scores[idx] * scale
            };
        }
    }
    write_f32(find(bindings, 0), &scores);
}

pub(crate) fn residual_add(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let residual = read_f32(find(bindings, 1));
    for (xi, ri) in x.iter_mut().zip(residual.iter()) {
        *xi += ri;
    }
    write_f32(find(bindings, 0), &x);
}

// dB[k,n] += sum_m A[m,k] * dC[m,n]   (A: MxK, dC: MxN, dB: KxN)
pub(crate) fn matmul_weight_bwd(bindings: &[CpuBinding]) {
    let a = read_f32(find(bindings, 0));
    let dc = read_f32(find(bindings, 1));
    let mut db = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 3));
    let (m, n, k) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    for kk in 0..k {
        for col in 0..n {
            let mut sum = 0.0f32;
            for row in 0..m {
                sum += a[row * k + kk] * dc[row * n + col];
            }
            db[kk * n + col] += sum;
        }
    }
    write_f32(find(bindings, 2), &db);
}

pub(crate) fn adamw(bindings: &[CpuBinding]) {
    let mut weights = read_f32(find(bindings, 0));
    let grads = read_f32(find(bindings, 1));
    let mut m = read_f32(find(bindings, 2));
    let mut v = read_f32(find(bindings, 3));
    let param_meta = read_u32(find(bindings, 4));
    let size = param_meta[0] as usize;

    let schedule = read_u32(find(bindings, 5));
    let step = schedule[0];
    let lr = f32::from_bits(schedule[1]);

    let const_cfg = read_u32(find(bindings, 6));
    let beta1 = f32::from_bits(const_cfg[0]);
    let beta2 = f32::from_bits(const_cfg[1]);
    let eps = f32::from_bits(const_cfg[2]);
    let weight_decay = f32::from_bits(const_cfg[3]);

    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);

    for idx in 0..size {
        let g = grads[idx];
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;

        let m_hat = m[idx] / bias_correction1;
        let v_hat = v[idx] / bias_correction2;

        let theta = weights[idx];
        weights[idx] = theta - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * theta);
    }

    write_f32(find(bindings, 0), &weights);
    write_f32(find(bindings, 2), &m);
    write_f32(find(bindings, 3), &v);
}

pub(crate) fn adamw_schedule(bindings: &[CpuBinding]) {
    let state = read_u32(find(bindings, 0));
    let step = state[0] + 1;

    let cfg = read_u32(find(bindings, 1));
    let lr_max = f32::from_bits(cfg[0]);
    let lr_min = f32::from_bits(cfg[1]);
    let warmup_steps = cfg[2];
    let max_steps = cfg[3];

    let lr = if step < warmup_steps {
        lr_max * step as f32 / warmup_steps as f32
    } else {
        let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
    };

    write_f32(find(bindings, 0), &[f32::from_bits(step), lr]);
}

pub(crate) fn zero_tensor(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 1));
    let len = meta[0] as usize;
    write_f32(find(bindings, 0), &vec![0.0f32; len]);
}
