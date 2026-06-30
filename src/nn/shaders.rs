pub enum BuiltInShader {
    // Forward
    MatMul,
    Embedding,
    CausalMask,
    SiLU,
    RoPE,
    Softmax,
    RMSNorm,
    ResidualAdd,
    CrossEntropy,
    HeadGather,
    HeadScatter,
    ZeroTensor,
    // Backward
    MatMulTrp,
    MatMulWeightBwd,
    SiLUBwd,
    RoPEBwd,
    SoftmaxBwd,
    RMSNormBwd,
    RMSNormWeightBwd,
    EmbeddingBwd,
    CrossEntropyBwd,
    BwdAddInplace,
    // Optimizer
    AdamW,
}

impl BuiltInShader {
    pub fn name(&self) -> &'static str {
        match self {
            Self::MatMul => "MatMul",
            Self::Embedding => "Embedding",
            Self::CausalMask => "CausalMask",
            Self::SiLU => "SiLU",
            Self::RoPE => "RoPE",
            Self::Softmax => "Softmax",
            Self::RMSNorm => "RMSNorm",
            Self::ResidualAdd => "ResidualAdd",
            Self::CrossEntropy => "CrossEntropy",
            Self::HeadGather => "HeadGather",
            Self::HeadScatter => "HeadScatter",
            Self::ZeroTensor => "ZeroTensor",
            Self::MatMulTrp => "MatMulTrp",
            Self::MatMulWeightBwd => "MatMulWeightBwd",
            Self::SiLUBwd => "SiLUBwd",
            Self::RoPEBwd => "RoPEBwd",
            Self::SoftmaxBwd => "SoftmaxBwd",
            Self::RMSNormBwd => "RMSNormBwd",
            Self::RMSNormWeightBwd => "RMSNormWeightBwd",
            Self::EmbeddingBwd => "EmbeddingBwd",
            Self::CrossEntropyBwd => "CrossEntropyBwd",
            Self::BwdAddInplace => "BwdAddInplace",
            Self::AdamW => "AdamW",
        }
    }
}
