use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, ShaderDef, TensorBind, TensorMode};
use wilupgu::tensor::Tensor;

#[test]
fn test_mixed_transformer_block() {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let matmul_def = ShaderDef::new(
        "MatMul",
        include_str!("../src/shaders/fwd/matmul.wgsl"),
        vec![
            TensorMode::Input,
            TensorMode::Input,
            TensorMode::Output,
            TensorMode::Meta,
        ],
    );
    let silu_def = ShaderDef::new(
        "SiLU",
        include_str!("../src/shaders/fwd/silu.wgsl"),
        vec![TensorMode::InOut],
    );
    let add_def = ShaderDef::new(
        "ResidualAdd",
        include_str!("../src/shaders/add.wgsl"),
        vec![TensorMode::InOut, TensorMode::Input],
    );

    let vec_size = 16;
    let input_data = vec![1.0f32; vec_size];
    let weight_data = vec![2.0f32; vec_size]; // Çarpınca hepsi 2.0 olacak
    let residual_data = vec![10.0f32; vec_size]; // Sonradan eklenecek
    let meta_data = vec![vec_size as u32, 1, 1];

    let t_input = Tensor::init_from_cpu(ctx.clone(), &input_data, wgpu::BufferUsages::STORAGE);
    let t_weight = Tensor::init_from_cpu(ctx.clone(), &weight_data, wgpu::BufferUsages::STORAGE);
    let t_residual =
        Tensor::init_from_cpu(ctx.clone(), &residual_data, wgpu::BufferUsages::STORAGE);
    let t_meta = Tensor::init_from_cpu(ctx.clone(), &meta_data, wgpu::BufferUsages::STORAGE);

    let t_main_output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Main_Output_Buf"),
        size: (vec_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let t_out = Tensor {
        ctx: ctx.clone(),
        buffer: t_main_output.into(),
        size: (vec_size * 4) as u64,
    };

    let mut graph = ComputeGraph::new(ctx.clone());

    // (1.0 * 2.0 = 2.0)
    graph.add_node(
        &matmul_def,
        &[
            TensorBind {
                binding: 0,
                tensor: &t_input,
                mode: TensorMode::Input,
            },
            TensorBind {
                binding: 1,
                tensor: &t_weight,
                mode: TensorMode::Input,
            },
            TensorBind {
                binding: 2,
                tensor: &t_out,
                mode: TensorMode::Output,
            },
            TensorBind {
                binding: 3,
                tensor: &t_meta,
                mode: TensorMode::Meta,
            },
        ],
        [16, 1, 1],
    );

    // 2.0 * sigmoid(2.0) ~ 1.761
    graph.add_node(
        &silu_def,
        &[TensorBind {
            binding: 0,
            tensor: &t_out,
            mode: TensorMode::InOut,
        }],
        [16, 1, 1],
    );

    // 1.761 + 10.0 ≈ 11.761
    graph.add_node(
        &add_def,
        &[
            TensorBind {
                binding: 0,
                tensor: &t_out,
                mode: TensorMode::InOut,
            },
            TensorBind {
                binding: 1,
                tensor: &t_residual,
                mode: TensorMode::Input,
            },
        ],
        [16, 1, 1],
    );

    graph.execute();

    let result: Vec<f32> = t_out.to_cpu();
    println!("--> Block Result: {:?}", &result[0..4]);

    // 11.7 ile 11.8 arası
    assert!(
        result[0] > 11.7 && result[0] < 11.8,
        "Crashed: Gelen değer: {}",
        result[0]
    );
}

//  TEST: İDİOT-PROOF TEST
// burası çökmeli ve panic atmalı. aşağıdaki kod panic atar ise başarılı.
#[test]
#[should_panic(expected = "Tensor Mode Mismatch")]
fn test_validation_idiot_proof() {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let silu_def = ShaderDef::new("SiLU", "...", vec![TensorMode::InOut]); // SiLU InOut bekler!

    let dummy_data = vec![1.0f32; 16];
    let t_dummy = Tensor::init_from_cpu(ctx.clone(), &dummy_data, wgpu::BufferUsages::STORAGE);

    let mut graph = ComputeGraph::new(ctx.clone());

    // yanlış TensorMode
    graph.add_node(
        &silu_def,
        &[TensorBind {
            binding: 0,
            tensor: &t_dummy,
            mode: TensorMode::Input,
        }],
        [16, 1, 1],
    );
}
