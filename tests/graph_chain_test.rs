use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, ShaderDef, TensorBind, TensorMode};
use wilupgu::tensor::Tensor;

#[test]
fn test_brutal_10_layer_chain() {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let matmul_def = ShaderDef::new(
        "ChainMatMul",
        include_str!("../src/shaders/fwd/matmul.wgsl"),
        vec![
            TensorMode::Input,
            TensorMode::Input,
            TensorMode::Output,
            TensorMode::Meta,
        ],
    );

    let vec_size = 32;
    let raw_data: Vec<f32> = vec![1.0; vec_size];
    let weight_data: Vec<f32> = vec![2.0; vec_size];
    let meta_data: Vec<u32> = vec![vec_size as u32, 1, 1];

    let meta_tensor = Tensor::init_from_cpu(ctx.clone(), &meta_data, wgpu::BufferUsages::STORAGE);
    let weight_tensor =
        Tensor::init_from_cpu(ctx.clone(), &weight_data, wgpu::BufferUsages::STORAGE);

    let mut chain_tensors: Vec<Tensor> = Vec::new();

    chain_tensors.push(Tensor::init_from_cpu(
        ctx.clone(),
        &raw_data,
        wgpu::BufferUsages::STORAGE,
    ));

    for _ in 0..10 {
        let empty_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chain_Out_Buf"),
            size: (vec_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        chain_tensors.push(Tensor {
            ctx: ctx.clone(),
            buffer: empty_buf.into(),
            size: (vec_size * 4) as u64,
        });
    }

    let mut graph = ComputeGraph::new(ctx.clone());

    for i in 0..10 {
        let (left_slice, right_slice) = chain_tensors.split_at(i + 1);
        let input_t = &left_slice[i];
        let output_t = &right_slice[0];

        graph.add_node(
            &matmul_def,
            &[
                TensorBind {
                    binding: 0,
                    tensor: input_t,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &weight_tensor,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: output_t,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &meta_tensor,
                    mode: TensorMode::Meta,
                },
            ],
            [32, 1, 1],
        );
    }

    graph.execute();

    let final_output: Vec<f32> = chain_tensors[10].to_cpu();

    println!("--> Son Katman Örneği: {:?}", &final_output[0..4]);
    assert_eq!(
        final_output[0], 1024.0,
        "Chain crashed: waited 1024.0, camed: {}",
        final_output[0]
    );
}
