use crate::graph::{ShaderDef, TensorMode};

pub enum BuiltInShader {
    MatMul,
    Embedding,
    CausalMask,
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
                "MatMul_L1_Tiled",
                include_str!("../shaders/matmul.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::Embedding => ShaderDef::new(
                "Embedding",
                include_str!("../shaders/embedding.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::CausalMask => ShaderDef::new(
                "CausalMask",
                include_str!("../shaders/causal_mask.wgsl"),
                vec![TensorMode::InOut, TensorMode::Meta],
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
