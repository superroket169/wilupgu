use crate::graph::{ShaderDef, TensorMode};

pub enum BuiltInShader {
    MatMul,
    RMSNorm,
    SiLU,
    RoPE,
    ResidualAdd,
    Softmax,
}

impl BuiltInShader {
    pub fn get_def(&self) -> ShaderDef {
        match self {
            Self::MatMul => ShaderDef::new(
                "MatMul",
                include_str!("../shaders/matmul.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::RMSNorm => ShaderDef::new(
                "RMSNorm",
                include_str!("../shaders/rmsnorm.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::SiLU => ShaderDef::new(
                "SiLU",
                include_str!("../shaders/silu.wgsl"),
                vec![TensorMode::InOut],
            ),
            Self::ResidualAdd => ShaderDef::new(
                "ResidualAdd",
                include_str!("../shaders/add.wgsl"),
                vec![TensorMode::InOut, TensorMode::Input],
            ),
            Self::Softmax => ShaderDef::new(
                "Softmax",
                include_str!("../shaders/softmax.wgsl"),
                vec![TensorMode::InOut, TensorMode::Meta],
            ),
            Self::RoPE => ShaderDef::new(
                "RoPE",
                include_str!("../shaders/rope.wgsl"),
                vec![TensorMode::InOut, TensorMode::Meta],
            ),
        }
    }
}
