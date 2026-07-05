pub(crate) mod cpu_kernels;
pub(crate) mod cuda_kernels;

use crate::backend::TensorMode::{InOut, Input, Meta, Output};
#[cfg(feature = "cuda")]
use crate::backends::cuda::dispatch as cd;
use crate::shader::{CudaShape, CudaSpec, Shader};
use cpu_kernels as cpu;
use cuda_kernels as k;

// ==========================================================================
//  Permanent built-ins
// ==========================================================================

pub static MATMUL: Shader = Shader {
    name: "MatMul",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("../shaders/fwd/matmul.wgsl")),
    cpu: Some(cpu::matmul),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cd::custom_matmul),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static MATMUL_TRP: Shader = Shader {
    name: "MatMulTrp",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("../shaders/fwd/matmul_trp.wgsl")),
    cpu: Some(cpu::matmul_trp),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cd::custom_matmul_trp),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static MATMUL_ADD: Shader = Shader {
    name: "MatMulAdd",
    layout: &[Input, Input, InOut, Meta],
    wgpu: Some(include_str!("../shaders/fwd/matmul_add.wgsl")),
    cpu: Some(cpu::matmul_add),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cd::custom_matmul_add),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static MATMUL_WEIGHT_BWD: Shader = Shader {
    name: "MatMulWeightBwd",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("../shaders/bwd/matmul_weight_trp.wgsl")),
    cpu: Some(cpu::matmul_weight_bwd),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cd::custom_matmul_weight_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static RESIDUAL_ADD: Shader = Shader {
    name: "ResidualAdd",
    layout: &[InOut, Input],
    wgpu: Some(include_str!("../shaders/add.wgsl")),
    cpu: Some(cpu::residual_add),
    cuda: Some(CudaSpec {
        src: k::ADD,
        entry: "add_kernel",
        shape: CudaShape::Add,
    }),
};

pub static BWD_ADD_INPLACE: Shader = Shader {
    name: "BwdAddInplace",
    layout: &[InOut, Input],
    wgpu: Some(include_str!("../shaders/bwd/bwd_add_inplace.wgsl")),
    cpu: Some(cpu::residual_add),
    cuda: Some(CudaSpec {
        src: k::BWD_ADD_INPLACE,
        entry: "bwd_add_inplace_kernel",
        shape: CudaShape::Add,
    }),
};

pub static ZERO_TENSOR: Shader = Shader {
    name: "ZeroTensor",
    layout: &[Output, Meta],
    wgpu: Some(include_str!("../shaders/zero_tensor.wgsl")),
    cpu: Some(cpu::zero_tensor),
    cuda: Some(CudaSpec {
        src: k::ZERO_TENSOR,
        entry: "zero_tensor_kernel",
        shape: CudaShape::InOut1,
    }),
};

pub static ADAMW: Shader = Shader {
    name: "AdamW",
    layout: &[InOut, Input, InOut, InOut, Meta, Meta],
    wgpu: Some(include_str!("../shaders/bwd/adamw.wgsl")),
    cpu: Some(cpu::adamw),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cd::custom_adamw),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static CAUSAL_MASK: Shader = Shader {
    name: "CausalMask",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("../shaders/causal_mask.wgsl")),
    cpu: Some(cpu::causal_mask),

    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cd::custom_causal_mask),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};
