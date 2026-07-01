// Backend parity test suite.
//
// Every kernel is run on WGPU and (when the `cuda` feature is enabled and a
// GPU is present) on CUDA.  Results are compared element-wise within a tight
// tolerance.  A CPU reference implementation is also provided for each op so
// that both backends are validated for *correctness*, not just mutual agreement.
//
//   cargo test --test backend_parity                   # WGPU only
//   cargo test --test backend_parity --features cuda   # WGPU + CUDA
//
// Design notes:
//   - Every `run_*` function is generic over Backend so the exact same code
//     path exercises both backends.
//   - CPU references are computed in plain Rust; no external deps.
//   - Tolerances are deliberately tight (FP32 rounding only); loosen a
//     constant if a future kernel legitimately needs more headroom.

use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

#[cfg(feature = "cuda")]
use wilupgu::CudaBackend;

// ── tolerances ────────────────────────────────────────────────────────────────

// Simple arithmetic (add, scale, zero).
const EPS_EXACT: f32 = 1e-5;
// Ops with a single exp/sqrt (silu, rmsnorm, causal-mask scale).
const EPS_TIGHT: f32 = 2e-4;
// Ops with softmax / row-wise reduction chains.
const EPS_NORM: f32 = 5e-4;
// Cross-entropy (log + exp in the same kernel).
const EPS_LOSS: f32 = 1e-3;

// ── comparison ────────────────────────────────────────────────────────────────

fn assert_close(label: &str, a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    let worst = a
        .iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| (i, (x - y).abs()))
        .max_by(|p, q| p.1.partial_cmp(&q.1).unwrap());
    if let Some((idx, err)) = worst {
        assert!(
            err <= eps,
            "{label}[{idx}]: a={:.6} b={:.6}  err={:.2e}  eps={:.2e}",
            a[idx],
            b[idx],
            err,
            eps
        );
    }
}

// ── backend init ──────────────────────────────────────────────────────────────

fn wgpu() -> Arc<WgpuBackend> {
    Arc::new(pollster::block_on(WgpuBackend::new()))
}

#[cfg(feature = "cuda")]
fn cuda_ctx() -> Option<Arc<CudaBackend>> {
    CudaBackend::new(0).ok().map(Arc::new)
}

// ── meta structs (must match the shader / CUDA kernel byte layout) ────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsMeta {
    seq_len: u32,
    size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScaleMeta {
    seq_len: u32,
    scale: f32,
}

// ── generic kernel runners ────────────────────────────────────────────────────

fn run_matmul<B: Backend>(ctx: Arc<B>, a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
    let ta = Tensor::init_from_cpu(ctx.clone(), a);
    let tb = Tensor::init_from_cpu(ctx.clone(), b);
    let tc = Tensor::new(ctx.clone(), (m * n * 4) as u64);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[m, n, k]);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "MatMul",
        &[
            Binding::new(0, &ta.buffer, TensorMode::Input),
            Binding::new(1, &tb.buffer, TensorMode::Input),
            Binding::new(2, &tc.buffer, TensorMode::Output),
            Binding::new(3, &tm.buffer, TensorMode::Meta),
        ],
        [(n + 15) / 16, (m + 15) / 16, 1],
    );
    g.execute();
    ctx.synchronize();
    tc.to_cpu()
}

fn run_matmul_trp<B: Backend>(
    ctx: Arc<B>,
    a: &[f32],
    b: &[f32],
    m: u32,
    n: u32,
    k: u32,
) -> Vec<f32> {
    let ta = Tensor::init_from_cpu(ctx.clone(), a);
    let tb = Tensor::init_from_cpu(ctx.clone(), b);
    let tc = Tensor::new(ctx.clone(), (m * n * 4) as u64);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[m, n, k]);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "MatMulTrp",
        &[
            Binding::new(0, &ta.buffer, TensorMode::Input),
            Binding::new(1, &tb.buffer, TensorMode::Input),
            Binding::new(2, &tc.buffer, TensorMode::Output),
            Binding::new(3, &tm.buffer, TensorMode::Meta),
        ],
        [(n + 15) / 16, (m + 15) / 16, 1],
    );
    g.execute();
    ctx.synchronize();
    tc.to_cpu()
}

fn run_silu<B: Backend>(ctx: Arc<B>, input: &[f32]) -> Vec<f32> {
    let n = input.len() as u32;
    let t = Tensor::init_from_cpu(ctx.clone(), input);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "SiLU",
        &[Binding::new(0, &t.buffer, TensorMode::InOut)],
        [(n + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    t.to_cpu()
}

fn run_silu_bwd<B: Backend>(ctx: Arc<B>, x_pre: &[f32], dy: &[f32]) -> Vec<f32> {
    let n = x_pre.len() as u32;
    let tx = Tensor::init_from_cpu(ctx.clone(), x_pre);
    let tdy = Tensor::init_from_cpu(ctx.clone(), dy);
    let tdx = Tensor::new(ctx.clone(), (n * 4) as u64);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "SiLUBwd",
        &[
            Binding::new(0, &tx.buffer, TensorMode::Input),
            Binding::new(1, &tdy.buffer, TensorMode::Input),
            Binding::new(2, &tdx.buffer, TensorMode::Output),
        ],
        [(n + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    tdx.to_cpu()
}

fn run_residual_add<B: Backend>(ctx: Arc<B>, x: &[f32], r: &[f32]) -> Vec<f32> {
    let n = x.len() as u32;
    let tx = Tensor::init_from_cpu(ctx.clone(), x);
    let tr = Tensor::init_from_cpu(ctx.clone(), r);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "ResidualAdd",
        &[
            Binding::new(0, &tx.buffer, TensorMode::InOut),
            Binding::new(1, &tr.buffer, TensorMode::Input),
        ],
        [(n + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    tx.to_cpu()
}

fn run_softmax<B: Backend>(ctx: Arc<B>, scores: &[f32], seq_len: u32) -> Vec<f32> {
    let ts = Tensor::init_from_cpu(ctx.clone(), scores);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[seq_len]);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "Softmax",
        &[
            Binding::new(0, &ts.buffer, TensorMode::InOut),
            Binding::new(1, &tm.buffer, TensorMode::Meta),
        ],
        [(seq_len + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    ts.to_cpu()
}

fn run_rmsnorm<B: Backend>(
    ctx: Arc<B>,
    input: &[f32],
    weight: &[f32],
    seq_len: u32,
    dim: u32,
) -> Vec<f32> {
    let ti = Tensor::init_from_cpu(ctx.clone(), input);
    let tw = Tensor::init_from_cpu(ctx.clone(), weight);
    let to = Tensor::new(ctx.clone(), (seq_len * dim * 4) as u64);
    let tm = Tensor::init_from_cpu(
        ctx.clone(),
        &[RmsMeta {
            seq_len,
            size: dim,
            eps: 1e-5,
        }],
    );
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "RMSNorm",
        &[
            Binding::new(0, &ti.buffer, TensorMode::Input),
            Binding::new(1, &tw.buffer, TensorMode::Input),
            Binding::new(2, &to.buffer, TensorMode::Output),
            Binding::new(3, &tm.buffer, TensorMode::Meta),
        ],
        [seq_len, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    to.to_cpu()
}

fn run_causal_mask<B: Backend>(ctx: Arc<B>, scores: &[f32], seq_len: u32, scale: f32) -> Vec<f32> {
    let ts = Tensor::init_from_cpu(ctx.clone(), scores);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[AttnScaleMeta { seq_len, scale }]);
    let grid = (seq_len + 15) / 16;
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "CausalMask",
        &[
            Binding::new(0, &ts.buffer, TensorMode::InOut),
            Binding::new(1, &tm.buffer, TensorMode::Meta),
        ],
        [grid, grid, 1],
    );
    g.execute();
    ctx.synchronize();
    ts.to_cpu()
}

fn run_zero_tensor<B: Backend>(ctx: Arc<B>, n: u32) -> Vec<f32> {
    let t = Tensor::init_from_cpu(ctx.clone(), &vec![1.0f32; n as usize]);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[n]);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "ZeroTensor",
        &[
            Binding::new(0, &t.buffer, TensorMode::Output),
            Binding::new(1, &tm.buffer, TensorMode::Meta),
        ],
        [(n + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    t.to_cpu()
}

fn run_cross_entropy<B: Backend>(
    ctx: Arc<B>,
    logits: &[f32],
    targets: &[u32],
    vocab: u32,
    rows: u32,
) -> (Vec<f32>, Vec<f32>) {
    let tl = Tensor::init_from_cpu(ctx.clone(), logits);
    // Targets are u32 but stored byte-identical in an f32 buffer.
    let tt: Vec<f32> = targets.iter().map(|&t| f32::from_bits(t)).collect();
    let tt = Tensor::init_from_cpu(ctx.clone(), &tt);
    let tp = Tensor::new(ctx.clone(), (rows * vocab * 4) as u64);
    let tlos = Tensor::new(ctx.clone(), (rows * 4) as u64);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[vocab, rows]);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        "CrossEntropy",
        &[
            Binding::new(0, &tl.buffer, TensorMode::Input),
            Binding::new(1, &tt.buffer, TensorMode::Input),
            Binding::new(2, &tp.buffer, TensorMode::Output),
            Binding::new(3, &tlos.buffer, TensorMode::Output),
            Binding::new(4, &tm.buffer, TensorMode::Meta),
        ],
        [(rows + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    (tp.to_cpu(), tlos.to_cpu())
}

// ── CPU reference implementations ─────────────────────────────────────────────

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        }
    }
    c
}

fn cpu_matmul_trp(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
        }
    }
    c
}

fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

fn cpu_silu_bwd(x: &[f32], dy: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(dy)
        .map(|(&xi, &dyi)| {
            let sig = 1.0 / (1.0 + (-xi).exp());
            dyi * sig * (1.0 + xi * (1.0 - sig))
        })
        .collect()
}

fn cpu_softmax_rows(scores: &[f32], seq_len: usize) -> Vec<f32> {
    let mut out = scores.to_vec();
    for row in out.chunks_mut(seq_len) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        for v in row.iter_mut() {
            *v = (*v - max).exp() / sum;
        }
    }
    out
}

fn cpu_rmsnorm(input: &[f32], weight: &[f32], seq_len: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * dim];
    for s in 0..seq_len {
        let row = &input[s * dim..(s + 1) * dim];
        let rms = (row.iter().map(|&v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        for d in 0..dim {
            out[s * dim + d] = weight[d] * row[d] / rms;
        }
    }
    out
}

fn cpu_cross_entropy(logits: &[f32], targets: &[u32], vocab: usize, rows: usize) -> Vec<f32> {
    let mut losses = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        let log_sum = max + sum.ln();
        losses[r] = log_sum - row[targets[r] as usize];
    }
    losses
}

// ── test helpers ──────────────────────────────────────────────────────────────

macro_rules! run_all_backends {
    ($runner:expr) => {{
        let wgpu_out: Vec<f32> = $runner(wgpu());

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda) = cuda_ctx() {
                let cuda_out: Vec<f32> = $runner(cuda);
                assert_close("backend parity", &wgpu_out, &cuda_out, EPS_NORM);
            }
        }

        wgpu_out
    }};
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn parity_matmul_known() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = cpu_matmul(&a, &b, 2, 2, 3);

    let out = run_all_backends!(|ctx| run_matmul(ctx, &a, &b, 2, 2, 3));
    assert_close("MatMul (known)", &out, &expected, EPS_TIGHT);
}

#[test]
fn parity_matmul_large() {
    let (m, n, k) = (16usize, 16usize, 32usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| {
            ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) % 1000)
                as f32
                * 0.01
        })
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as u64).wrapping_mul(2891336453).wrapping_add(1) % 1000) as f32 * 0.01 - 0.5)
        .collect();
    let expected = cpu_matmul(&a, &b, m, n, k);

    let out = run_all_backends!(|ctx| { run_matmul(ctx, &a, &b, m as u32, n as u32, k as u32) });
    assert_close("MatMul (large)", &out, &expected, EPS_TIGHT);
}

#[test]
fn parity_matmul_trp() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3] → B^T = first two cols of I_3
    let expected = cpu_matmul_trp(&a, &b, 2, 2, 3);

    let out = run_all_backends!(|ctx| run_matmul_trp(ctx, &a, &b, 2, 2, 3));
    assert_close("MatMulTrp", &out, &expected, EPS_TIGHT);
}

#[test]
fn parity_silu() {
    let input: Vec<f32> = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
    let expected = cpu_silu(&input);

    let out = run_all_backends!(|ctx| run_silu(ctx, &input));
    assert_close("SiLU", &out, &expected, EPS_TIGHT);
}

#[test]
fn parity_silu_bwd() {
    let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, -0.5];
    let dy = vec![1.0f32; x.len()];
    let expected = cpu_silu_bwd(&x, &dy);

    let out = run_all_backends!(|ctx| run_silu_bwd(ctx, &x, &dy));
    assert_close("SiLUBwd", &out, &expected, EPS_TIGHT);
}

#[test]
fn parity_silu_bwd_nonuniform_grad() {
    let x: Vec<f32> = vec![-1.0, 0.3, 1.7, -2.5, 4.0, -0.1, 0.0, 2.2];
    let dy: Vec<f32> = vec![0.1, -0.2, 0.5, 1.0, -1.0, 0.3, 0.7, -0.4];
    let expected = cpu_silu_bwd(&x, &dy);

    let out = run_all_backends!(|ctx| run_silu_bwd(ctx, &x, &dy));
    assert_close("SiLUBwd (non-uniform grad)", &out, &expected, EPS_TIGHT);
}

#[test]
fn parity_residual_add() {
    let x: Vec<f32> = vec![1.0, -1.0, 2.5, 0.0, -3.0, 7.0, 0.1, 100.0, -50.0, 0.001];
    let r: Vec<f32> = vec![0.5, 0.5, -0.5, 1.0, 3.0, -7.0, 0.9, -100.0, 50.0, 999.0];
    let expected: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();

    let out = run_all_backends!(|ctx| run_residual_add(ctx, &x, &r));
    assert_close("ResidualAdd", &out, &expected, EPS_EXACT);
}

#[test]
fn parity_softmax_basic() {
    let seq_len = 4u32;
    let scores: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4, -1.0, 0.0, 1.0, 2.0, 10.0, 1.0, 0.1, 0.01,
    ];
    let expected = cpu_softmax_rows(&scores, seq_len as usize);

    let out = run_all_backends!(|ctx| run_softmax(ctx, &scores, seq_len));
    assert_close("Softmax", &out, &expected, EPS_NORM);

    for (r, chunk) in out.chunks(seq_len as usize).enumerate() {
        let s: f32 = chunk.iter().sum();
        assert!((s - 1.0).abs() < EPS_NORM, "Softmax row {r} sum={s}");
        assert!(chunk.iter().all(|&v| v >= 0.0 && v <= 1.0 + EPS_NORM));
    }
}

#[test]
fn parity_softmax_numerical_stability() {
    let seq_len = 4u32;
    let scores: Vec<f32> = vec![
        1000.0, 1001.0, 999.0, 998.0, -500.0, -501.0, -502.0, -503.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 1.0,
    ];
    let expected = cpu_softmax_rows(&scores, seq_len as usize);

    let out = run_all_backends!(|ctx| run_softmax(ctx, &scores, seq_len));
    assert_close("Softmax (extreme)", &out, &expected, EPS_NORM);
}

#[test]
fn parity_rmsnorm_unit_weight() {
    let seq_len = 4u32;
    let dim = 8u32;
    let input: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight = vec![1.0f32; dim as usize];
    let expected = cpu_rmsnorm(&input, &weight, seq_len as usize, dim as usize, 1e-5);

    let out = run_all_backends!(|ctx| run_rmsnorm(ctx, &input, &weight, seq_len, dim));
    assert_close("RMSNorm (unit weight)", &out, &expected, EPS_NORM);

    // With unit weight each output row should have RMS ≈ 1.
    for (r, chunk) in out.chunks(dim as usize).enumerate() {
        let rms = (chunk.iter().map(|v| v * v).sum::<f32>() / dim as f32).sqrt();
        assert!((rms - 1.0).abs() < EPS_NORM, "RMSNorm row {r} rms={rms}");
    }
}

#[test]
fn parity_rmsnorm_nonunit_weight() {
    let seq_len = 3u32;
    let dim = 6u32;
    let input: Vec<f32> = vec![
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, -0.5, 1.5, -1.5,
        2.0,
    ];
    let weight: Vec<f32> = vec![2.0, 0.5, 1.0, -1.0, 0.3, 1.5];
    let expected = cpu_rmsnorm(&input, &weight, seq_len as usize, dim as usize, 1e-5);

    let out = run_all_backends!(|ctx| run_rmsnorm(ctx, &input, &weight, seq_len, dim));
    assert_close("RMSNorm (non-unit weight)", &out, &expected, EPS_NORM);
}

#[test]
fn parity_causal_mask() {
    let seq_len = 4u32;
    let scale = 0.25f32;
    let scores = vec![1.0f32; (seq_len * seq_len) as usize];

    let out = run_all_backends!(|ctx| run_causal_mask(ctx, &scores, seq_len, scale));

    for i in 0..seq_len as usize {
        for j in 0..seq_len as usize {
            let v = out[i * seq_len as usize + j];
            if j > i {
                assert!(
                    v < -1e8,
                    "CausalMask[{i},{j}] upper-tri should be -inf, got {v}"
                );
            } else {
                assert!(
                    (v - scale).abs() < EPS_EXACT,
                    "CausalMask[{i},{j}]: got={v} want={scale}"
                );
            }
        }
    }
}

#[test]
fn parity_causal_mask_larger() {
    let seq_len = 8u32;
    let scale = 1.0 / (64.0f32).sqrt(); // typical attention scale for head_dim=64

    let scores: Vec<f32> = (0..(seq_len * seq_len) as usize)
        .map(|i| i as f32 * 0.1)
        .collect();

    let out = run_all_backends!(|ctx| run_causal_mask(ctx, &scores, seq_len, scale));

    for i in 0..seq_len as usize {
        for j in 0..seq_len as usize {
            let idx = i * seq_len as usize + j;
            let v = out[idx];
            if j > i {
                assert!(v < -1e8, "CausalMask({i},{j}) upper-tri = {v}");
            } else {
                let want = scores[idx] * scale;
                assert!(
                    (v - want).abs() < EPS_EXACT,
                    "CausalMask({i},{j}): got={v} want={want}"
                );
            }
        }
    }
}

#[test]
fn parity_zero_tensor() {
    let out = run_all_backends!(|ctx| run_zero_tensor(ctx, 512));
    assert!(
        out.iter().all(|&v| v == 0.0),
        "ZeroTensor produced non-zero values"
    );
}

#[test]
fn parity_cross_entropy_basic() {
    let vocab = 4u32;
    let rows = 3u32;
    let logits: Vec<f32> = vec![1.0, 3.0, 2.0, 0.5, 0.1, 0.2, 5.0, 0.3, 2.0, 1.0, 1.0, 1.0];
    let targets: Vec<u32> = vec![1, 2, 0];
    let expected_losses = cpu_cross_entropy(&logits, &targets, vocab as usize, rows as usize);

    let (_probs, losses) = {
        let logits = logits.clone();
        let targets = targets.clone();
        let wgpu_out = run_cross_entropy(wgpu(), &logits, &targets, vocab, rows);

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda) = cuda_ctx() {
                let cuda_out = run_cross_entropy(cuda, &logits, &targets, vocab, rows);
                assert_close(
                    "CrossEntropy losses parity",
                    &wgpu_out.1,
                    &cuda_out.1,
                    EPS_LOSS,
                );
                assert_close(
                    "CrossEntropy probs parity",
                    &wgpu_out.0,
                    &cuda_out.0,
                    EPS_LOSS,
                );
            }
        }
        wgpu_out
    };

    assert_close("CrossEntropy losses", &losses, &expected_losses, EPS_LOSS);

    for (r, chunk) in _probs.chunks(vocab as usize).enumerate() {
        let s: f32 = chunk.iter().sum();
        assert!(
            (s - 1.0).abs() < EPS_LOSS,
            "CrossEntropy probs row {r} sum={s}"
        );
    }
}

#[test]
fn parity_cross_entropy_uniform_logits() {
    let vocab = 8u32;
    let rows = 4u32;
    let logits = vec![1.0f32; (vocab * rows) as usize];
    let targets: Vec<u32> = vec![0, 3, 7, 2];
    let expected_loss = (vocab as f32).ln(); // ≈ 2.079

    let (_probs, losses) = {
        let wgpu_out = run_cross_entropy(wgpu(), &logits, &targets, vocab, rows);

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda) = cuda_ctx() {
                let cuda_out = run_cross_entropy(cuda, &logits, &targets, vocab, rows);
                assert_close(
                    "CrossEntropy (uniform) parity",
                    &wgpu_out.1,
                    &cuda_out.1,
                    EPS_LOSS,
                );
            }
        }
        wgpu_out
    };

    for (r, &loss) in losses.iter().enumerate() {
        assert!(
            (loss - expected_loss).abs() < EPS_LOSS,
            "CrossEntropy uniform row {r}: got={loss} want={expected_loss}"
        );
    }
}

// ── full pipeline smoke: MatMul → SiLU → ResidualAdd ─────────────────────────

fn run_ffn_block<B: Backend>(
    ctx: Arc<B>,
    x: &[f32],
    w1: &[f32],
    residual: &[f32],
    m: u32,
    n: u32,
    k: u32,
) -> Vec<f32> {
    let tx = Tensor::init_from_cpu(ctx.clone(), x);
    let tw = Tensor::init_from_cpu(ctx.clone(), w1);
    let tr = Tensor::init_from_cpu(ctx.clone(), residual);
    let hidden = Tensor::new(ctx.clone(), (m * n * 4) as u64);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[m, n, k]);

    let mut g = ComputeGraph::new(ctx.clone());
    // MatMul
    g.add_node(
        "MatMul",
        &[
            Binding::new(0, &tx.buffer, TensorMode::Input),
            Binding::new(1, &tw.buffer, TensorMode::Input),
            Binding::new(2, &hidden.buffer, TensorMode::Output),
            Binding::new(3, &tm.buffer, TensorMode::Meta),
        ],
        [(n + 15) / 16, (m + 15) / 16, 1],
    );
    // SiLU
    g.add_node(
        "SiLU",
        &[Binding::new(0, &hidden.buffer, TensorMode::InOut)],
        [(m * n + 255) / 256, 1, 1],
    );
    // ResidualAdd
    g.add_node(
        "ResidualAdd",
        &[
            Binding::new(0, &hidden.buffer, TensorMode::InOut),
            Binding::new(1, &tr.buffer, TensorMode::Input),
        ],
        [(m * n + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    hidden.to_cpu()
}

#[test]
fn parity_ffn_pipeline() {
    let (m, n, k) = (4u32, 4u32, 4u32);
    let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let w: Vec<f32> = (0..16)
        .map(|i| if i % 5 == 0 { 1.0 } else { 0.1 })
        .collect();
    let residual = vec![0.5f32; 16];

    let out = run_all_backends!(|ctx| run_ffn_block(ctx, &x, &w, &residual, m, n, k));

    assert!(
        out.iter().all(|v| v.is_finite()),
        "FFN pipeline produced non-finite values"
    );
    assert!(
        out.iter().any(|&v| v != 0.0),
        "FFN pipeline produced all-zero output"
    );

    for &v in &out {
        assert!(v > -1.0, "FFN pipeline: suspiciously negative value {v}");
    }
}
