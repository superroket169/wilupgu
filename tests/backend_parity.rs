// Backend parity test suite.

use std::sync::Arc;
use wilupgu::builtin;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

#[cfg(feature = "cuda")]
use wilupgu::CudaBackend;

// ── tolerances ────────────────────────────────────────────────────────────────

// Simple arithmetic (add, scale, zero).
const EPS_EXACT: f32 = 1e-5;
// Ops with a single exp/sqrt (silu, rmsnorm, causal-mask scale).
const EPS_TIGHT: f32 = 2e-4;
// wgpu/cuda cross-backend comparison tolerance (only used when `cuda` is on).
#[cfg(feature = "cuda")]
const EPS_ABS: f32 = 1e-4;
#[cfg(feature = "cuda")]
const EPS_REL: f32 = 5e-3;

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

#[cfg(feature = "cuda")]
fn assert_close_rel(label: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
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
        .map(|(i, (x, y))| {
            let err = (x - y).abs();
            let tol = atol + rtol * x.abs().max(y.abs());
            (i, err, tol)
        })
        .max_by(|p, q| (p.1 - p.2).partial_cmp(&(q.1 - q.2)).unwrap());
    if let Some((idx, err, tol)) = worst {
        assert!(
            err <= tol,
            "{label}[{idx}]: a={:.6} b={:.6}  err={:.2e}  tol={:.2e} (atol={:.1e} rtol={:.1e})",
            a[idx],
            b[idx],
            err,
            tol,
            atol,
            rtol
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
        &builtin::MATMUL,
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
        &builtin::MATMUL_TRP,
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

fn run_residual_add<B: Backend>(ctx: Arc<B>, x: &[f32], r: &[f32]) -> Vec<f32> {
    let n = x.len() as u32;
    let tx = Tensor::init_from_cpu(ctx.clone(), x);
    let tr = Tensor::init_from_cpu(ctx.clone(), r);
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        &builtin::RESIDUAL_ADD,
        &[
            Binding::new(0, &tx.buffer, TensorMode::Accumulate),
            Binding::new(1, &tr.buffer, TensorMode::Input),
        ],
        [(n + 255) / 256, 1, 1],
    );
    g.execute();
    ctx.synchronize();
    tx.to_cpu()
}

fn run_causal_mask<B: Backend>(ctx: Arc<B>, scores: &[f32], seq_len: u32, scale: f32) -> Vec<f32> {
    let ts = Tensor::init_from_cpu(ctx.clone(), scores);
    let tm = Tensor::init_from_cpu(ctx.clone(), &[AttnScaleMeta { seq_len, scale }]);
    let grid = (seq_len + 15) / 16;
    let mut g = ComputeGraph::new(ctx.clone());
    g.add_node(
        &builtin::CAUSAL_MASK,
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
        &builtin::ZERO_TENSOR,
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

// ── test helpers ──────────────────────────────────────────────────────────────

macro_rules! run_all_backends {
    ($runner:expr) => {{
        let wgpu_out: Vec<f32> = $runner(wgpu());

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda) = cuda_ctx() {
                let cuda_out: Vec<f32> = $runner(cuda);
                assert_close_rel("backend parity", &wgpu_out, &cuda_out, EPS_ABS, EPS_REL);
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
            ((i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                % 1000) as f32
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
fn parity_residual_add() {
    let x: Vec<f32> = vec![1.0, -1.0, 2.5, 0.0, -3.0, 7.0, 0.1, 100.0, -50.0, 0.001];
    let r: Vec<f32> = vec![0.5, 0.5, -0.5, 1.0, 3.0, -7.0, 0.9, -100.0, 50.0, 999.0];
    let expected: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();

    let out = run_all_backends!(|ctx| run_residual_add(ctx, &x, &r));
    assert_close("ResidualAdd", &out, &expected, EPS_EXACT);
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
