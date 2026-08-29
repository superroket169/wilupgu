# Shader kataloğu

Her shader'ın wgsl/cuda/cpu/meta/emitter parçalarını tek satırda görmek için.
Kaynak: wilupgu `src/builtin/mod.rs` (12 builtin) + akasha-core
`src/shaders/mod.rs` (30 shader), 2026-08-29 itibarıyla. Yeni shader
eklerken buraya bir satır ekle (bkz. ARCHITECTURE.md "Yeni kernel ekleme
checklist'i" madde 8). Bu dosya iki repo'yu birlikte kapsar çünkü akasha
sadece wilupgu'nun builtin'lerini tüketen bir downstream'dir — tek kaynaktan
okumak daha doğru.

## wilupgu builtin'leri (12) — `src/builtin/mod.rs`

| Shader | wgsl | cuda | cpu | Kullanan |
|---|---|---|---|---|
| MatMul | fwd/matmul.wgsl | Custom → cuBLAS | ✓ | akasha `matmul()` (m>1) |
| Gemv | fwd/gemv.wgsl | Custom → cuBLAS | ✓ | akasha `matmul_with()` (m=1, H6 auto-route) |
| GemvAdd | fwd/gemv_add.wgsl | Custom → cuBLAS | ✓ | akasha `matmul_add_with()` (m=1) |
| MatMulTrp | fwd/matmul_trp.wgsl | Custom → cuBLAS | ✓ | akasha `matmul_trp()` |
| MatMulAdd | fwd/matmul_add.wgsl | Custom → cuBLAS | ✓ | akasha `matmul_add_with()` (m>1, block_pre_attn/block_post_attn own the meta); plain `matmul_add()` is now `#[cfg(test)]`-only |
| MatMulWeightBwd | bwd/matmul_weight_trp.wgsl | Custom → cuBLAS | ✓ | akasha `matmul_weight_bwd()` |
| ResidualAdd | add.wgsl | Generic (`ADD`) | ✓ | akasha `residual_add()` |
| BwdAddInplace | bwd/bwd_add_inplace.wgsl | Generic (`BWD_ADD_INPLACE`) | ✓ | akasha `add_inplace_bwd()` |
| ZeroTensor | zero_tensor.wgsl | Generic (`ZERO_TENSOR`) | ✓ | akasha `zero()` |
| AdamW | bwd/adamw.wgsl | Custom → `launch_adamw` | ✓ | akasha `optim/adamw.rs` |
| AdamWSchedule | bwd/adamw_schedule.wgsl | Custom → `launch_adamw_schedule` | ✓ | akasha `optim/adamw.rs` |
| CausalMask | causal_mask.wgsl | Generic (`CAUSAL_MASK`) | ✓ | **hiçbir akasha çağıran yok** — sadece wilupgu'nun kendi `backend_parity` testi kullanıyor; H5'te causal attention flash'a taşınınca akasha tarafı bırakılmış olmalı. ember'da kullanılıyor olabilir, kontrol edilmedi |

Beşi (matmul ailesi) CUDA'da cuBLAS'a gidiyor — B9/decode-cuBLAS'sızlaştırma
tartışmasının konusu tam bunlar arasından `Gemv`/`GemvAdd` (m=1).

## akasha-core shader'ları (30) — `src/shaders/mod.rs`

| Shader | wgsl | cuda src | cpu | Meta struct | Emitter (`ops/emit.rs`) |
|---|---|---|---|---|---|
| Embedding | fwd/embedding.wgsl | ✓ | ✓ | EmbeddingMeta | `embedding()` |
| EmbeddingBwd | bwd/embedding_bwd.wgsl | ✓ | ✓ | EmbeddingMeta | `embedding_bwd()` |
| SiLU | fwd/silu.wgsl | ✓ | ✓ | — | `silu()` |
| SiLUOut | fwd/silu_out.wgsl | ✓ | ✓ | — | `silu_out()` |
| Add | fwd/add.wgsl | ✓ | ✓ | — | `add_out()` |
| SiLUBwd | bwd/silu_bwd.wgsl | ✓ | **yok** | — | `silu_bwd()` |
| RoPE | fwd/rope.wgsl | ✓ | ✓ | RopeMeta | `rope()` |
| RoPEBwd | bwd/rope_bwd.wgsl | ✓ | **yok** | RopeMeta | `rope_bwd()` — **`#[cfg(test)]` only**, sadece `rope_qk` fusion testinin referansı, production'da hiç çağrılmıyor |
| RopeQK | fwd/rope_qk.wgsl | ✓ | **yok** | RopeMeta | `rope_qk()` — gerçek yol (fused rope_bwd+rope_bwd'nin production karşılığı) |
| RopeBwdQK | bwd/rope_bwd_qk.wgsl | ✓ | **yok** | RopeMeta | `rope_bwd_qk()` |
| RoPEOffset | fwd/rope_offset.wgsl | ✓ | ✓ | RopeOffsetMeta | `rope_offset_with()` (decode) |
| AttnQkCached | fwd/attn_qk_cached.wgsl | ✓ | ✓ | AttnCachedMeta | `attn_qk_cached_with()` (decode) |
| AttnAvCached | fwd/attn_av_cached.wgsl | ✓ | ✓ | AttnCachedMeta | `attn_av_cached_with()` (decode) |
| SoftmaxRect | fwd/softmax_rect.wgsl | ✓ | ✓ | SoftmaxRectMeta | `softmax_rect_with()` (decode) |
| RMSNorm | fwd/rmsnorm.wgsl | ✓ | ✓ | NormMeta | `rmsnorm()` |
| RMSNormBwd | bwd/rmsnorm_bwd.wgsl | ✓ | **yok** | NormMeta | `rmsnorm_bwd()` |
| RMSNormWeightBwd | bwd/rmsnorm_weight_bwd.wgsl | ✓ | **yok** | NormMeta | `rmsnorm_bwd()` (aynı emitter, ikinci node) |
| CrossEntropy | fwd/cross_entropy.wgsl | ✓ | ✓ | CrossEntropyMeta | `cross_entropy()` |
| CrossEntropyBwd | bwd/cross_entropy_bwd.wgsl | ✓ | ✓ | CrossEntropyMeta | `cross_entropy_bwd()` |
| HeadGather | head_gather.wgsl | ✓ | ✓ | HeadMoveMeta | `head_gather()` / `head_gather_with()` |
| HeadScatter | head_scatter.wgsl | ✓ | ✓ | HeadMoveMeta | `head_scatter()` — **`#[cfg(test)]` only**, sadece `qkv_scatter` fusion testinin referansı |
| QkvSplit | fwd/qkv_split.wgsl | ✓ | **yok** | HeadMoveMeta | `qkv_split()` |
| QkvScatter | bwd/qkv_scatter.wgsl | ✓ | **yok** | HeadMoveMeta | `qkv_scatter()` |
| FlashAttention | fwd/flash_attention.wgsl | ✓ | **yok** | FlashAttnMeta | `flash_attention()` |
| FlashAttentionBwdDQ | bwd/flash_attention_bwd_dq.wgsl | ✓ | **yok** | FlashAttnMeta | `flash_attention_bwd()` |
| FlashAttentionBwdDKDV | bwd/flash_attention_bwd_dkdv.wgsl | ✓ | **yok** | FlashAttnMeta | `flash_attention_bwd()` (aynı emitter, ikinci node) |
| CacheWrite | cache_write.wgsl | ✓ | ✓ | CacheWriteMeta | `cache_write()` / `cache_write_with()` |
| GradSumSq | bwd/grad_sumsq.wgsl | ✓ | **yok** | GradSumSqMeta | `grad_sumsq()` |
| GradNormScale | bwd/grad_norm_scale.wgsl | ✓ | **yok** | GradNormMeta | `grad_norm_scale()` |
| GradScale | bwd/grad_scale.wgsl | ✓ | **yok** | ZeroMeta | `grad_scale()` |

`ZeroMeta` ayrıca wilupgu'nun `ZeroTensor`/akasha'nın `zero()` emitter'ında da
kullanılıyor (`len` dışında alan taşımayan tüm kernellerin ortak metası).

## Notlar

- **CPU boşlukları (14/30 akasha shader'ı):** SiLUBwd, RoPEBwd, RopeQK,
  RopeBwdQK, RMSNormBwd, RMSNormWeightBwd, QkvSplit, QkvScatter,
  FlashAttention + iki bwd'si, GradSumSq, GradNormScale, GradScale. Hepsi
  bwd/perf-kritik tarafta — CPU backend şu an tam bir eğitim döngüsünü
  koşturamaz, yalnız fwd-ağırlıklı testleri. ARCHITECTURE.md "Fikir
  kuyruğu → test mimarisi" maddesindeki "birkaç bwd kernel'inin CPU impl'i
  yok" notu, tam liste haline geldi.
- **Ölü/test-only iki shader:** `RoPEBwd` ve `HeadScatter` gerçek hiçbir
  fwd/bwd yolunda çağrılmıyor — yalnız `rope_qk`/`qkv_scatter` fusion
  testlerinin bağımsız referans implementasyonu olarak yaşıyorlar
  (`#[cfg(test)]`). Bilinçli, ARCHITECTURE.md'de zaten dokümante ("rope_bwd
  ve head_scatter yalnız #[cfg(test)] yaşar").
- **wilupgu'da tek ölü builtin:** `CausalMask` — akasha hiç çağırmıyor,
  yalnız wilupgu'nun kendi parity testinde var.
- Meta yok (`—`) demek kernel'in Meta binding'i olmadığı, boyutun yalnız
  grid'den geldiği anlamına gelir (akasha tablosunda SiLU/SiLUOut/Add).
  `ZeroTensor`/`GradScale`'in ikisi de aynı `ZeroMeta{len}`'i paylaşır.
