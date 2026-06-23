use crate::graph::{ShaderDef, TensorMode};

pub enum BuiltInShader {
    // --- FORWARD PASS ---
    MatMul,
    Embedding,
    CausalMask,
    SiLU,
    RoPE,
    Softmax,
    RMSNorm,
    ResidualAdd,
    CrossEntropy,

    // --- BACKWARD PASS  ---
    MatMulTrp,
    MatMulWeightBwd,
    SiLUBwd,
    RoPEBwd,
    SoftmaxBwd,
    RMSNormBwd,
    RMSNormWeightBwd,
    EmbeddingBwd,
    CrossEntropyBwd,

    // --- OPTIMIZER ---
    AdamW,
}

impl BuiltInShader {
    pub fn get_def(&self) -> ShaderDef {
        match self {
            // FORWARD
            Self::MatMul => ShaderDef::new(
                "MatMul",
                include_str!("../shaders/fwd/matmul.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::Embedding => ShaderDef::new(
                "Embedding",
                include_str!("../shaders/fwd/embedding.wgsl"),
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
            Self::SiLU => ShaderDef::new(
                "SiLU",
                include_str!("../shaders/fwd/silu.wgsl"),
                vec![TensorMode::InOut],
            ),
            Self::RoPE => ShaderDef::new(
                "RoPE",
                include_str!("../shaders/fwd/rope.wgsl"),
                vec![TensorMode::InOut, TensorMode::Meta],
            ),
            Self::Softmax => ShaderDef::new(
                "Softmax",
                include_str!("../shaders/fwd/softmax.wgsl"),
                vec![TensorMode::InOut, TensorMode::Meta],
            ),
            Self::RMSNorm => ShaderDef::new(
                "RMSNorm",
                include_str!("../shaders/fwd/rmsnorm.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::ResidualAdd => ShaderDef::new(
                "ResidualAdd",
                include_str!("../shaders/add.wgsl"),
                vec![TensorMode::InOut, TensorMode::Input],
            ),
            Self::CrossEntropy => ShaderDef::new(
                "CrossEntropy",
                include_str!("../shaders/fwd/cross_entropy.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),

            // BACKWARD
            Self::MatMulTrp => ShaderDef::new(
                "MatMulTrp",
                include_str!("../shaders/fwd/matmul_trp.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::MatMulWeightBwd => ShaderDef::new(
                "MatMulWeightBwd",
                include_str!("../shaders/bwd/matmul_weight_trp.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::SiLUBwd => ShaderDef::new(
                "SiLUBwd",
                include_str!("../shaders/bwd/silu_bwd.wgsl"),
                vec![TensorMode::Input, TensorMode::Input, TensorMode::Output],
            ),
            Self::RoPEBwd => ShaderDef::new(
                "RoPEBwd",
                include_str!("../shaders/bwd/rope_bwd.wgsl"),
                vec![TensorMode::InOut, TensorMode::Meta],
            ),
            Self::SoftmaxBwd => ShaderDef::new(
                "SoftmaxBwd",
                include_str!("../shaders/bwd/softmax_bwd.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::RMSNormBwd => ShaderDef::new(
                "RMSNormBwd",
                include_str!("../shaders/bwd/rmsnorm_bwd.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::RMSNormWeightBwd => ShaderDef::new(
                "RMSNormWeightBwd",
                include_str!("../shaders/bwd/rmsnorm_weight_bwd.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::EmbeddingBwd => ShaderDef::new(
                "EmbeddingBwd",
                include_str!("../shaders/bwd/embedding_bwd.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),
            Self::CrossEntropyBwd => ShaderDef::new(
                "CrossEntropyBwd",
                include_str!("../shaders/bwd/cross_entropy_bwd.wgsl"),
                vec![
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Input,
                    TensorMode::Output,
                    TensorMode::Meta,
                ],
            ),

            // OPTIMIZER
            Self::AdamW => ShaderDef::new(
                "AdamW",
                include_str!("../shaders/bwd/adamw.wgsl"),
                vec![
                    TensorMode::InOut,
                    TensorMode::Input,
                    TensorMode::InOut,
                    TensorMode::InOut,
                    TensorMode::Meta,
                    TensorMode::Meta,
                ],
            ),
        }
    }
}
