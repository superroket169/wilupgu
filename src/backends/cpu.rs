use crate::backend::{Backend, Binding};
use crate::pool::BufferPool;
use std::sync::{Arc, Mutex};

pub type CpuBuffer = Arc<Mutex<Vec<u8>>>;

#[derive(Clone)]
struct CpuBinding {
    slot: u32,
    buffer: CpuBuffer,
}

#[derive(Clone)]
pub struct CpuNode {
    name: String,
    bindings: Vec<CpuBinding>,
}

pub struct CpuBackend {
    pool: BufferPool<CpuBuffer>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            pool: BufferPool::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

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
//            Kernel implementations
// ========================================

// C[row, col] = sum_k A[row, k] * B[k, col]   (A: MxK, B: KxN, C: MxN)
fn matmul(bindings: &[CpuBinding]) {
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
fn matmul_trp(bindings: &[CpuBinding]) {
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
fn matmul_add(bindings: &[CpuBinding]) {
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

fn embedding(bindings: &[CpuBinding]) {
    let tokens = read_u32(find(bindings, 0));
    let weight = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (vocab_size, embed_dim, seq_len) = (meta[0], meta[1] as usize, meta[2] as usize);

    let mut out = vec![0.0f32; seq_len * embed_dim];
    for t in 0..seq_len {
        let token_id = tokens[t];
        if token_id < vocab_size {
            let w_off = token_id as usize * embed_dim;
            let o_off = t * embed_dim;
            out[o_off..o_off + embed_dim].copy_from_slice(&weight[w_off..w_off + embed_dim]);
        }
    }
    write_f32(find(bindings, 2), &out);
}

fn causal_mask(bindings: &[CpuBinding]) {
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

fn silu(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
    write_f32(find(bindings, 0), &x);
}

fn rope(bindings: &[CpuBinding]) {
    let mut vec_ = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (seq_len, dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let offset = token_idx * dim + h * head_dim + dim_idx;
                let x0 = vec_[offset];
                let x1 = vec_[offset + 1];

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = token_idx as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                vec_[offset] = x0 * v_cos - x1 * v_sin;
                vec_[offset + 1] = x0 * v_sin + x1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &vec_);
}

fn rope_offset(bindings: &[CpuBinding]) {
    let mut vec_ = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (seq_len, dim, head_dim, pos_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let abs_pos = token_idx + pos_offset;
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let offset = token_idx * dim + h * head_dim + dim_idx;
                let x0 = vec_[offset];
                let x1 = vec_[offset + 1];

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = abs_pos as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                vec_[offset] = x0 * v_cos - x1 * v_sin;
                vec_[offset + 1] = x0 * v_sin + x1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &vec_);
}

fn softmax(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let seq_len = meta[0] as usize;

    for row in 0..seq_len {
        let off = row * seq_len;
        let max_val = x[off..off + seq_len]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for i in 0..seq_len {
            let e = (x[off + i] - max_val).exp();
            x[off + i] = e;
            sum_exp += e;
        }
        for i in 0..seq_len {
            x[off + i] /= sum_exp;
        }
    }
    write_f32(find(bindings, 0), &x);
}

fn causal_softmax(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let seq_len = meta[0] as usize;
    let scale = f32::from_bits(meta[1]);

    fn masked(x: &[f32], off: usize, row: usize, i: usize, scale: f32) -> f32 {
        if i > row {
            -1_000_000_000.0
        } else {
            x[off + i] * scale
        }
    }

    for row in 0..seq_len {
        let off = row * seq_len;
        let max_val = (0..seq_len)
            .map(|i| masked(&x, off, row, i, scale))
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for i in 0..seq_len {
            let e = (masked(&x, off, row, i, scale) - max_val).exp();
            x[off + i] = e;
            sum_exp += e;
        }
        for i in 0..seq_len {
            x[off + i] /= sum_exp;
        }
    }
    write_f32(find(bindings, 0), &x);
}

fn softmax_rect(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (num_rows, width) = (meta[0] as usize, meta[1] as usize);
    let scale = f32::from_bits(meta[2]);

    for row in 0..num_rows {
        let off = row * width;
        let max_val = x[off..off + width]
            .iter()
            .map(|v| v * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for i in 0..width {
            let e = (x[off + i] * scale - max_val).exp();
            x[off + i] = e;
            sum_exp += e;
        }
        for i in 0..width {
            x[off + i] /= sum_exp;
        }
    }
    write_f32(find(bindings, 0), &x);
}

fn rmsnorm(bindings: &[CpuBinding]) {
    let x = read_f32(find(bindings, 0));
    let weight = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (seq_len, size) = (meta[0] as usize, meta[1] as usize);
    let eps = f32::from_bits(meta[2]);

    let mut out = vec![0.0f32; seq_len * size];
    for row in 0..seq_len {
        let off = row * size;
        let ss: f32 = x[off..off + size].iter().map(|v| v * v).sum();
        let rsqrt = 1.0 / ((ss / size as f32) + eps).sqrt();
        for i in 0..size {
            out[off + i] = x[off + i] * rsqrt * weight[i];
        }
    }
    write_f32(find(bindings, 2), &out);
}

fn residual_add(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let residual = read_f32(find(bindings, 1));
    for (xi, ri) in x.iter_mut().zip(residual.iter()) {
        *xi += ri;
    }
    write_f32(find(bindings, 0), &x);
}

fn cache_write(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let mut dst = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (row_count, width, dst_row_offset) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    for row in 0..row_count {
        let src_off = row * width;
        let dst_off = (dst_row_offset + row) * width;
        dst[dst_off..dst_off + width].copy_from_slice(&src[src_off..src_off + width]);
    }
    write_f32(find(bindings, 1), &dst);
}

fn head_gather(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 2));
    let (seq_len, full_dim, head_dim, head_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );

    let mut dst = read_f32(find(bindings, 1));
    for row in 0..seq_len {
        let src_off = row * full_dim + head_offset;
        let dst_off = row * head_dim;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
    write_f32(find(bindings, 1), &dst);
}

fn head_scatter(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let mut dst = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (seq_len, full_dim, head_dim, head_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );

    for row in 0..seq_len {
        let src_off = row * head_dim;
        let dst_off = row * full_dim + head_offset;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
    write_f32(find(bindings, 1), &dst);
}

fn cross_entropy(bindings: &[CpuBinding]) {
    let logits = read_f32(find(bindings, 0));
    let targets = read_u32(find(bindings, 1));
    let meta = read_u32(find(bindings, 4));
    let (vocab_size, num_rows) = (meta[0] as usize, meta[1] as usize);

    let mut probs = vec![0.0f32; num_rows * vocab_size];
    let mut losses = vec![0.0f32; num_rows];

    for row in 0..num_rows {
        let off = row * vocab_size;
        let target_id = targets[row] as usize;

        let max_val = logits[off..off + vocab_size]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        for i in 0..vocab_size {
            let e = (logits[off + i] - max_val).exp();
            probs[off + i] = e;
            sum_exp += e;
        }
        for i in 0..vocab_size {
            probs[off + i] /= sum_exp;
        }

        let log_sum_exp = sum_exp.ln();
        losses[row] = -(logits[off + target_id] - max_val - log_sum_exp);
    }

    write_f32(find(bindings, 2), &probs);
    write_f32(find(bindings, 3), &losses);
}

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;
    type Node = CpuNode;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn alloc(&self, size_bytes: u64) -> CpuBuffer {
        if let Some(buf) = self.pool.take(size_bytes) {
            return buf;
        }
        Arc::new(Mutex::new(vec![0u8; size_bytes as usize]))
    }

    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> CpuBuffer {
        let bytes = bytemuck::cast_slice(data);
        if let Some(buf) = self.pool.take(bytes.len() as u64) {
            buf.lock().unwrap().copy_from_slice(bytes);
            return buf;
        }
        Arc::new(Mutex::new(bytes.to_vec()))
    }

    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &CpuBuffer, data: &[T]) {
        buf.lock()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(data));
    }

    fn copy_to_cpu<T: bytemuck::Pod + Default + Clone>(&self, buf: &CpuBuffer) -> Vec<T> {
        let g = buf.lock().unwrap();
        bytemuck::cast_slice::<u8, T>(&g).to_vec()
    }

    fn free_buffer(&self, _buf: CpuBuffer) {}

    fn recycle(&self, size_bytes: u64, buf: CpuBuffer) {
        self.pool.recycle(size_bytes, buf);
    }

    fn is_sole_owner(buf: &CpuBuffer) -> bool {
        Arc::strong_count(buf) == 1
    }

    fn build_node(
        &self,
        kernel: &str,
        bindings: &[Binding<CpuBuffer>],
        _workgroups: [u32; 3],
    ) -> CpuNode {
        CpuNode {
            name: kernel.to_string(),
            bindings: bindings
                .iter()
                .map(|b| CpuBinding {
                    slot: b.slot,
                    buffer: b.buffer.clone(),
                })
                .collect(),
        }
    }

    fn execute(&self, nodes: &[CpuNode]) {
        for node in nodes {
            let b = &node.bindings;
            match node.name.as_str() {
                "MatMul" => matmul(b),
                "MatMulTrp" => matmul_trp(b),
                "MatMulAdd" => matmul_add(b),
                "Embedding" => embedding(b),
                "CausalMask" => causal_mask(b),
                "SiLU" => silu(b),
                "RoPE" => rope(b),
                "RoPEOffset" => rope_offset(b),
                "Softmax" => softmax(b),
                "SoftmaxRect" => softmax_rect(b),
                "CausalSoftmax" => causal_softmax(b),
                "RMSNorm" => rmsnorm(b),
                "ResidualAdd" => residual_add(b),
                "HeadGather" => head_gather(b),
                "HeadScatter" => head_scatter(b),
                "CrossEntropy" => cross_entropy(b),
                "CacheWrite" => cache_write(b),
                other => panic!("[cpu] unsupported kernel: {other}"),
            }
        }
    }

    fn synchronize(&self) {}
}
