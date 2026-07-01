//! Run with: cargo test --release --features cuda --test meta_cache_check

#![cfg(feature = "cuda")]

use std::sync::Arc;
use wilupgu::{Binding, ComputeGraph, CudaBackend, Tensor, TensorMode};

fn cuda_ctx() -> Arc<CudaBackend> {
    Arc::new(
        CudaBackend::new(0).expect("CUDA backend unavailable -- this check requires a CUDA GPU"),
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsMeta {
    seq_len: u32,
    size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamMeta {
    size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepConfig {
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

#[test]
fn meta_cache_check() {
    let ctx = cuda_ctx();

    // ---------------------------------------------------------------

    let seq_len: u32 = 2;
    let size: u32 = 4;
    let x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0];
    let w_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    let x = Tensor::init_from_cpu(ctx.clone(), &x_data);
    let w = Tensor::init_from_cpu(ctx.clone(), &w_data);
    let out = Tensor::init_from_cpu(ctx.clone(), &vec![0.0f32; 8]);
    let meta = Tensor::init_from_cpu(
        ctx.clone(),
        &[RmsMeta {
            seq_len,
            size,
            eps: 1e-5,
        }],
    );

    let mut graph = ComputeGraph::new(ctx.clone());
    graph.add_node(
        "RMSNorm",
        &[
            Binding::new(0, &x.buffer, TensorMode::Input),
            Binding::new(1, &w.buffer, TensorMode::Input),
            Binding::new(2, &out.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [seq_len, 1, 1],
    );

    graph.execute();
    ctx.synchronize();
    let run1: Vec<f32> = out.to_cpu();

    graph.execute();
    ctx.synchronize();
    let run2: Vec<f32> = out.to_cpu();

    assert_eq!(
        run1, run2,
        "RMSNorm output differed between two consecutive execute() calls!"
    );

    let expected: Vec<f32> = x_data
        .chunks(size as usize)
        .flat_map(|row| {
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
            let scale = 1.0 / (mean_sq + 1e-5).sqrt();
            row.iter().map(move |v| v * scale).collect::<Vec<_>>()
        })
        .collect();

    for (a, b) in run1.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "RMSNorm result {a} differs from expected {b}"
        );
    }

    println!("[check 1] RMSNorm: two consecutive execute() calls -> identical, correct output. OK");
    println!("  run1 = {:?}", run1);
    println!("  run2 = {:?}", run2);

    // ---------------------------------------------------------------

    let n: u32 = 4;
    let weight = Tensor::init_from_cpu(ctx.clone(), &[1.0f32, 1.0, 1.0, 1.0]);
    let grad = Tensor::init_from_cpu(ctx.clone(), &[1.0f32, 1.0, 1.0, 1.0]);
    let m = Tensor::init_from_cpu(ctx.clone(), &[0.0f32; 4]);
    let v = Tensor::init_from_cpu(ctx.clone(), &[0.0f32; 4]);
    let param_meta = Tensor::init_from_cpu(ctx.clone(), &[ParamMeta { size: n }]);
    let cfg = Tensor::init_from_cpu(
        ctx.clone(),
        &[StepConfig {
            step: 0,
            lr: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }],
    );

    let mut adamw_graph = ComputeGraph::new(ctx.clone());
    adamw_graph.add_node(
        "AdamW",
        &[
            Binding::new(0, &weight.buffer, TensorMode::InOut),
            Binding::new(1, &grad.buffer, TensorMode::Input),
            Binding::new(2, &m.buffer, TensorMode::InOut),
            Binding::new(3, &v.buffer, TensorMode::InOut),
            Binding::new(4, &param_meta.buffer, TensorMode::Meta),
            Binding::new(5, &cfg.buffer, TensorMode::Meta),
        ],
        [(n + 255) / 256, 1, 1],
    );

    let lrs = [0.1f32, 0.01, 0.001];
    let mut weight_snapshots = Vec::new();
    for (i, &lr) in lrs.iter().enumerate() {
        cfg.copy_from_cpu(&[StepConfig {
            step: (i + 1) as u32,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }]);
        adamw_graph.execute();
        ctx.synchronize();
        weight_snapshots.push(weight.to_cpu::<f32>());
    }

    println!(
        "[check 2] AdamW weight after each step with lr = {:?}:",
        lrs
    );
    for (lr, w) in lrs.iter().zip(weight_snapshots.iter()) {
        println!("  lr={lr} -> weight={:?}", w);
    }

    let delta = |a: &Vec<f32>, b: &Vec<f32>| (a[0] - b[0]).abs();
    let d1 = (1.0f32 - weight_snapshots[0][0]).abs();
    let d2 = delta(&weight_snapshots[0], &weight_snapshots[1]);
    let d3 = delta(&weight_snapshots[1], &weight_snapshots[2]);

    println!("  per-step deltas: d1={d1:.6} (lr=0.1), d2={d2:.6} (lr=0.01), d3={d3:.6} (lr=0.001)");

    assert!(
        d1 > d2 * 3.0,
        "step1->2 delta did not shrink as expected when lr dropped 10x (cfg may be stale/cached)"
    );
    assert!(
        d2 > d3 * 3.0,
        "step2->3 delta did not shrink as expected when lr dropped 10x (cfg may be stale/cached)"
    );

    println!(
        "[check 2] AdamW: per-step deltas shrink in line with falling lr -> cfg_meta is read LIVE each dispatch, not cached. OK"
    );

    println!("\nAll checks passed.");
}
