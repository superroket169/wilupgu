use crate::graph::{ShaderDef, TensorMode};
mod shaders;

pub fn load_core_shaders() -> Vec<ShaderDef> {
    vec![
        ShaderDef::new(
            "MatMul",
            include_str!("../shaders/fwd/matmul.wgsl"),
            vec![
                TensorMode::Input,
                TensorMode::Input,
                TensorMode::InOut,
                TensorMode::Meta,
            ],
        ),
        ShaderDef::new(
            "RMSNorm",
            include_str!("../shaders/fwd/rmsnorm.wgsl"),
            vec![
                TensorMode::Input,
                TensorMode::Input,
                TensorMode::InOut,
                TensorMode::Meta,
            ],
        ),
        ShaderDef::new(
            "SiLU",
            include_str!("../shaders/fwd/silu.wgsl"),
            vec![TensorMode::InOut],
        ),
        ShaderDef::new(
            "RoPE",
            include_str!("../shaders/fwd/rope.wgsl"),
            vec![TensorMode::InOut, TensorMode::Meta],
        ),
    ]
}
