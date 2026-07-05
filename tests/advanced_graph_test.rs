use std::sync::Arc;
use wilupgu::builtin;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

fn wgpu() -> Arc<WgpuBackend> {
    Arc::new(pollster::block_on(WgpuBackend::new()))
}

fn run_mixed_transformer_block<B: Backend>(
    ctx: Arc<B>,
    input_data: &[f32],
    weight_data: &[f32],
    residual_data: &[f32],
    vec_size: u32,
) -> Vec<f32> {
    let meta_data = vec![vec_size, 1, 1];

    let t_input = Tensor::init_from_cpu(ctx.clone(), input_data);
    let t_weight = Tensor::init_from_cpu(ctx.clone(), weight_data);
    let t_residual = Tensor::init_from_cpu(ctx.clone(), residual_data);
    let t_meta = Tensor::init_from_cpu(ctx.clone(), &meta_data);
    let t_out = Tensor::new(ctx.clone(), (vec_size * 4) as u64);

    let mut graph = ComputeGraph::new(ctx.clone());

    // (1.0 * 2.0 = 2.0)
    graph.add_node(
        &builtin::MATMUL,
        &[
            Binding::new(0, &t_input.buffer, TensorMode::Input),
            Binding::new(1, &t_weight.buffer, TensorMode::Input),
            Binding::new(2, &t_out.buffer, TensorMode::Output),
            Binding::new(3, &t_meta.buffer, TensorMode::Meta),
        ],
        [16, 1, 1],
    );

    // 2.0 + 10.0 = 12.0
    graph.add_node(
        &builtin::RESIDUAL_ADD,
        &[
            Binding::new(0, &t_out.buffer, TensorMode::InOut),
            Binding::new(1, &t_residual.buffer, TensorMode::Input),
        ],
        [16, 1, 1],
    );

    graph.execute();
    ctx.synchronize();
    t_out.to_cpu()
}

#[test]
fn test_mixed_transformer_block() {
    let vec_size = 16u32;
    let input_data = vec![1.0f32; vec_size as usize];
    let weight_data = vec![2.0f32; vec_size as usize];
    let residual_data = vec![10.0f32; vec_size as usize];

    let result =
        run_mixed_transformer_block(wgpu(), &input_data, &weight_data, &residual_data, vec_size);
    println!("--> Block Result: {:?}", &result[0..4]);

    assert!(
        (result[0] - 12.0).abs() < 1e-4,
        "Crashed: Gelen değer: {}",
        result[0]
    );
}

//  TEST: İDİOT-PROOF TEST
// burası çökmeli ve panic atmalı. aşağıdaki kod panic atar ise başarılı.

#[test]
#[should_panic(expected = "Tensor Mode Mismatch")]
fn test_validation_idiot_proof() {
    let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

    let dummy_data = vec![1.0f32; 16];
    let t_dummy = Tensor::init_from_cpu(ctx.clone(), &dummy_data);

    let mut graph = ComputeGraph::new(ctx.clone());

    graph.add_node(
        &builtin::RESIDUAL_ADD,
        &[
            Binding::new(0, &t_dummy.buffer, TensorMode::Input),
            Binding::new(1, &t_dummy.buffer, TensorMode::Input),
        ],
        [16, 1, 1],
    );
}
