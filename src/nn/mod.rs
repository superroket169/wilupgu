use crate::graph::{ShaderDef, TensorMode};
mod shaders;

pub fn load_core_shaders() -> Vec<ShaderDef> {
    vec![
        ShaderDef::new(
            "MatMul",
            include_str!("../shaders/matmul.wgsl"),
            vec![
                TensorMode::Input,
                TensorMode::Input,
                TensorMode::InOut,
                TensorMode::Meta,
            ],
        ),
        ShaderDef::new(
            "RMSNorm",
            include_str!("../shaders/rmsnorm.wgsl"),
            vec![
                TensorMode::Input,
                TensorMode::Input,
                TensorMode::InOut,
                TensorMode::Meta,
            ],
        ),
        ShaderDef::new(
            "SiLU",
            include_str!("../shaders/silu.wgsl"),
            vec![TensorMode::InOut],
        ),
        ShaderDef::new(
            "RoPE",
            include_str!("../shaders/rope.wgsl"),
            vec![TensorMode::InOut, TensorMode::Meta],
        ),
    ]
}
