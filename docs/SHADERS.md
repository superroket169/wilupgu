# Shader kataloğu

Kaynak: wilupgu `src/builtin/mod.rs` (refactor sırasında silindi, builtin'ler
geri döndüğünde bu tablo güncellenecek).

| Shader | wgsl | cuda | cpu | Kullanan |
|---|---|---|---|---|
| MatMul | fwd/matmul.wgsl | Custom → cuBLAS | ✓ | sequexa `matmul()` (m>1) |
| Gemv | fwd/gemv.wgsl | Custom → cuBLAS | ✓ | sequexa `matmul_with()` (m=1, H6 auto-route) |
| GemvAdd | fwd/gemv_add.wgsl | Custom → cuBLAS | ✓ | sequexa `matmul_add_with()` (m=1) |
| MatMulTrp | fwd/matmul_trp.wgsl | Custom → cuBLAS | ✓ | sequexa `matmul_trp()` |
| MatMulAdd | fwd/matmul_add.wgsl | Custom → cuBLAS | ✓ | sequexa `matmul_add_with()` (m>1, block_pre_attn/block_post_attn own the meta); plain `matmul_add()` is now `#[cfg(test)]`-only |
| MatMulWeightBwd | bwd/matmul_weight_trp.wgsl | Custom → cuBLAS | ✓ | sequexa `matmul_weight_bwd()` |
| ResidualAdd | add.wgsl | Generic (`ADD`) | ✓ | sequexa `residual_add()` |
| BwdAddInplace | bwd/bwd_add_inplace.wgsl | Generic (`BWD_ADD_INPLACE`) | ✓ | sequexa `add_inplace_bwd()` |
| ZeroTensor | zero_tensor.wgsl | Generic (`ZERO_TENSOR`) | ✓ | sequexa `zero()` |
| AdamW | bwd/adamw.wgsl | Custom → `launch_adamw` | ✓ | sequexa `optim/adamw.rs` |
| AdamWSchedule | bwd/adamw_schedule.wgsl | Custom → `launch_adamw_schedule` | ✓ | sequexa `optim/adamw.rs` |
| CausalMask | causal_mask.wgsl | Generic (`CAUSAL_MASK`) | ✓ | sequexa çağırmıyor (causal attention flash'a taşındı) |
