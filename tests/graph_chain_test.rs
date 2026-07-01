use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

fn wgpu() -> Arc<WgpuBackend> {
    Arc::new(pollster::block_on(WgpuBackend::new()))
}

fn run_brutal_10_layer_chain<B: Backend>(ctx: Arc<B>, vec_size: u32) -> Vec<f32> {
    let raw_data: Vec<f32> = vec![1.0; vec_size as usize];
    let weight_data: Vec<f32> = vec![2.0; vec_size as usize];
    let meta_data: Vec<u32> = vec![vec_size, 1, 1];

    let meta_tensor = Tensor::init_from_cpu(ctx.clone(), &meta_data);
    let weight_tensor = Tensor::init_from_cpu(ctx.clone(), &weight_data);

    let mut chain_tensors: Vec<Tensor<B>> = Vec::new();
    chain_tensors.push(Tensor::init_from_cpu(ctx.clone(), &raw_data));

    for _ in 0..10 {
        chain_tensors.push(Tensor::new(ctx.clone(), (vec_size * 4) as u64));
    }

    let mut graph = ComputeGraph::new(ctx.clone());

    for i in 0..10 {
        let (left_slice, right_slice) = chain_tensors.split_at(i + 1);
        let input_t = &left_slice[i];
        let output_t = &right_slice[0];

        graph.add_node(
            "MatMul",
            &[
                Binding::new(0, &input_t.buffer, TensorMode::Input),
                Binding::new(1, &weight_tensor.buffer, TensorMode::Input),
                Binding::new(2, &output_t.buffer, TensorMode::Output),
                Binding::new(3, &meta_tensor.buffer, TensorMode::Meta),
            ],
            [32, 1, 1],
        );
    }

    graph.execute();
    ctx.synchronize();
    chain_tensors[10].to_cpu()
}

#[test]
fn test_brutal_10_layer_chain() {
    let vec_size = 32u32;
    let final_output = run_brutal_10_layer_chain(wgpu(), vec_size);

    println!("--> Son Katman Örneği: {:?}", &final_output[0..4]);
    assert_eq!(
        final_output[0], 1024.0,
        "Chain crashed: waited 1024.0, camed: {}",
        final_output[0]
    );
}
