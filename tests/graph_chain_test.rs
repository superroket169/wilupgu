use std::sync::Arc;
use wilupgu::builtin;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

fn wgpu() -> Arc<WgpuBackend> {
    Arc::new(pollster::block_on(WgpuBackend::new()))
}

// 10 chained MATMUL nodes, each one a column-vector scale: A is [m,1],
// B is [1,1] = 2.0, so layer output[row] = input[row] * 2. Distinct
// per-row inputs make a dead row (a grid that is too small) show up as a
// wrong value instead of hiding behind identical neighbours.
fn run_brutal_10_layer_chain<B: Backend>(ctx: Arc<B>, m: u32) -> Vec<f32> {
    let raw_data: Vec<f32> = (1..=m).map(|i| i as f32).collect();
    let weight_data: Vec<f32> = vec![2.0];
    let meta_data: Vec<u32> = vec![m, 1, 1]; // M, N, K

    let meta_tensor = Tensor::init_from_cpu(ctx.clone(), &meta_data);
    let weight_tensor = Tensor::init_from_cpu(ctx.clone(), &weight_data);

    let mut chain_tensors: Vec<Tensor<B>> = Vec::new();
    chain_tensors.push(Tensor::init_from_cpu(ctx.clone(), &raw_data));

    for _ in 0..10 {
        chain_tensors.push(Tensor::new(ctx.clone(), (m * 4) as u64));
    }

    let mut graph = ComputeGraph::new(ctx.clone());

    for i in 0..10 {
        let (left_slice, right_slice) = chain_tensors.split_at(i + 1);
        let input_t = &left_slice[i];
        let output_t = &right_slice[0];

        // MATMUL runs 16x16 threads per workgroup, x = col, y = row: the
        // grid must span all m rows. The old test passed [32, 1, 1] and
        // silently left rows 16..32 uncomputed.
        graph.add_node(
            &builtin::MATMUL,
            &[
                Binding::new(0, &input_t.buffer, TensorMode::Input),
                Binding::new(1, &weight_tensor.buffer, TensorMode::Input),
                Binding::new(2, &output_t.buffer, TensorMode::Output),
                Binding::new(3, &meta_tensor.buffer, TensorMode::Meta),
            ],
            [1, (m + 15) / 16, 1],
        );
    }

    graph.execute();
    ctx.synchronize();
    chain_tensors[10].to_cpu()
}

#[test]
fn test_brutal_10_layer_chain() {
    let m = 32u32;
    let final_output = run_brutal_10_layer_chain(wgpu(), m);

    // Every row must carry its own scaled value: (row+1) * 2^10.
    for (row, &got) in final_output.iter().enumerate() {
        let expected = (row + 1) as f32 * 1024.0;
        assert_eq!(
            got, expected,
            "row {row}: waited {expected}, camed: {got} -- rows beyond the \
             first workgroup were probably never dispatched"
        );
    }
}
