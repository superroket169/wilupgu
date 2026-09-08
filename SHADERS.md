# Shader kataloğu

Her shader'ın wgsl/cuda/cpu/meta/emitter parçalarını tek satırda görmek için.
Kaynak: wilupgu `src/builtin/mod.rs` (12 builtin) + sequexa-core
`src/shaders/mod.rs` (31 shader), 2026-09-08 itibarıyla. Yeni shader
eklerken buraya bir satır ekle (bkz. ARCHITECTURE.md "Yeni kernel ekleme
checklist'i" madde 8). Bu dosya iki repo'yu birlikte kapsar çünkü sequexa
sadece wilupgu'nun builtin'lerini tüketen bir downstream'dir — tek kaynaktan
okumak daha doğru.

## wilupgu builtin'leri (12) — `src/builtin/mod.rs`

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
| CausalMask | causal_mask.wgsl | Generic (`CAUSAL_MASK`) | ✓ | sequexa çağırmıyor (H5'te causal attention flash'a taşındı) — wilupgu'nun kendi `backend_parity` testinde builtin olarak yaşıyor |

Beşi (matmul ailesi) CUDA'da cuBLAS'a gidiyor — B9/decode-cuBLAS'sızlaştırma
tartışmasının konusu tam bunlar arasından `Gemv`/`GemvAdd` (m=1).

## sequexa-core shader'ları (30) — `src/shaders/mod.rs`

| Shader | wgsl | cuda src | cpu | Meta struct | Emitter (`ops/emit.rs`) |
|---|---|---|---|---|---|
| Embedding | fwd/embedding.wgsl | ✓ | ✓ | EmbeddingMeta | `embedding()` |
| EmbeddingBwd | bwd/embedding_bwd.wgsl | ✓ | ✓ | EmbeddingMeta | `embedding_bwd()` |
| SiLU | fwd/silu.wgsl | ✓ | ✓ | — | `silu()` |
| SiLUOut | fwd/silu_out.wgsl | ✓ | ✓ | — | `silu_out()` |
| Add | fwd/add.wgsl | ✓ | ✓ | — | `add_out()` |
| SiLUBwd | bwd/silu_bwd.wgsl | ✓ | ✓ | — | `silu_bwd()` |
| RoPE | fwd/rope.wgsl | ✓ | ✓ | RopeMeta | `rope()` |
| RoPEBwd | bwd/rope_bwd.wgsl | ✓ | ✓ | RopeMeta | `rope_bwd()` — **`#[cfg(test)]` only**, sadece `rope_qk` fusion testinin referansı, production'da hiç çağrılmıyor |
| RopeQK | fwd/rope_qk.wgsl | ✓ | ✓ | RopeMeta | `rope_qk()` — gerçek yol (fused rope_bwd+rope_bwd'nin production karşılığı) |
| RopeBwdQK | bwd/rope_bwd_qk.wgsl | ✓ | ✓ | RopeMeta | `rope_bwd_qk()` |
| RoPEOffset | fwd/rope_offset.wgsl | ✓ | ✓ | RopeOffsetMeta | `rope_offset_with()` (decode) |
| AttnQkCached | fwd/attn_qk_cached.wgsl | ✓ | ✓ | AttnCachedMeta | `attn_qk_cached_with()` (decode) |
| AttnAvCached | fwd/attn_av_cached.wgsl | ✓ | ✓ | AttnCachedMeta | `attn_av_cached_with()` (decode) |
| SoftmaxRect | fwd/softmax_rect.wgsl | ✓ | ✓ | SoftmaxRectMeta | `softmax_rect_with()` (decode) |
| RMSNorm | fwd/rmsnorm.wgsl | ✓ | ✓ | NormMeta | `rmsnorm()` |
| RMSNormBwd | bwd/rmsnorm_bwd.wgsl | ✓ | ✓ | NormMeta | `rmsnorm_bwd()` |
| RMSNormWeightBwd | bwd/rmsnorm_weight_bwd.wgsl | ✓ | ✓ | NormMeta | `rmsnorm_bwd()` (aynı emitter, ikinci node) |
| CrossEntropy | fwd/cross_entropy.wgsl | ✓ | ✓ | CrossEntropyMeta | `cross_entropy()` |
| CrossEntropyBwd | bwd/cross_entropy_bwd.wgsl | ✓ | ✓ | CrossEntropyMeta | `cross_entropy_bwd()` |
| HeadGather | head_gather.wgsl | ✓ | ✓ | HeadMoveMeta | `head_gather()` / `head_gather_with()` |
| HeadScatter | head_scatter.wgsl | ✓ | ✓ | HeadMoveMeta | `head_scatter()` — **`#[cfg(test)]` only**, sadece `qkv_scatter` fusion testinin referansı |
| QkvSplit | fwd/qkv_split.wgsl | ✓ | ✓ | HeadMoveMeta | `qkv_split()` |
| QkvScatter | bwd/qkv_scatter.wgsl | ✓ | ✓ | HeadMoveMeta | `qkv_scatter()` |
| FlashAttention | fwd/flash_attention.wgsl | ✓ | ✓ | FlashAttnMeta | `flash_attention()` |
| FlashAttentionBwdD | bwd/flash_attention_bwd_d.wgsl | ✓ | ✓ | FlashAttnMeta | `flash_attention_bwd()` — D[i] precompute (B11b), ilk node |
| FlashAttentionBwdDQ | bwd/flash_attention_bwd_dq.wgsl | ✓ | ✓ | FlashAttnMeta | `flash_attention_bwd()` |
| FlashAttentionBwdDKDV | bwd/flash_attention_bwd_dkdv.wgsl | ✓ | ✓ | FlashAttnMeta | `flash_attention_bwd()` (aynı emitter, son node) |
| CacheWrite | cache_write.wgsl | ✓ | ✓ | CacheWriteMeta | `cache_write()` / `cache_write_with()` |
| GradSumSq | bwd/grad_sumsq.wgsl | ✓ | ✓ | GradSumSqMeta | `grad_sumsq()` |
| GradNormScale | bwd/grad_norm_scale.wgsl | ✓ | ✓ | GradNormMeta | `grad_norm_scale()` |
| GradScale | bwd/grad_scale.wgsl | ✓ | ✓ | ZeroMeta | `grad_scale()` |

`ZeroMeta` ayrıca wilupgu'nun `ZeroTensor`/sequexa'nın `zero()` emitter'ında da
kullanılıyor (`len` dışında alan taşımayan tüm kernellerin ortak metası).
Meta yok (`—`) demek kernel'in Meta binding'i olmadığı, boyutun yalnız
grid'den geldiği anlamına gelir (sequexa tablosunda SiLU/SiLUOut/Add).
