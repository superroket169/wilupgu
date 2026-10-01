# Shader kataloğu

Kaynak kodlar `src/shader-codes/{wgsl,native,cuda}/`, her shader kendi
dosyasında.

## Çekirdek lineer cebir

| Shader | wgsl | cuda | native (cpu) |
|---|---|---|---|
| MatMul | matmul.wgsl | Custom → cuBLAS (henüz yok) | matmul.rs |
| Gemv | gemv.wgsl | Custom → cuBLAS (henüz yok) | gemv.rs |
| GemvAdd | gemv_add.wgsl | Custom → cuBLAS (henüz yok) | gemv_add.rs |
| MatMulTrp | matmul_trp.wgsl | Custom → cuBLAS (henüz yok) | matmul_trp.rs |
| MatMulAdd | matmul_add.wgsl | Custom → cuBLAS (henüz yok) | matmul_add.rs |
| Transpose | transpose.wgsl | — | transpose.rs |
| Dot | dot.wgsl | — | dot.rs |

## Elementwise

| Shader | wgsl | cuda | native (cpu) |
|---|---|---|---|
| Add | add.wgsl | add.cu | add.rs |
| AddInplace | add_inplace.wgsl | add_inplace.cu | add_inplace.rs |
| Mul | mul.wgsl | — | mul.rs |
| Scale | scale.wgsl | — | scale.rs |
| Max | max.wgsl | — | max.rs |
| Min | min.wgsl | — | min.rs |
| Clamp | clamp.wgsl | — | clamp.rs |

## Reduction

| Shader | wgsl | cuda | native (cpu) |
|---|---|---|---|
| Sum | sum.wgsl (çok-geçişli: workgroup başı bir partial, >1 partial kalırsa tekrar çağrılır) | — | sum.rs (tek geçiş) |
| Dot | yukarıda -- ilk geçişi kendi, geri kalanı `Sum`'la biter | — | yukarıda |

## Buffer init

| Shader | wgsl | cuda | native (cpu) |
|---|---|---|---|
| ZeroTensor | zero_tensor.wgsl | zero_tensor.cu | zero_tensor.rs |
| FillConstant | fill_constant.wgsl | — | fill_constant.rs |
| FillRandom | fill_random.wgsl (hash tabanlı, kriptografik değil) | — | fill_random.rs (aynı hash, CPU/GPU parity için) |

## Kapsam dışına alınanlar

wilupgu'dan çıkarıldı:

- **AdamW**, **AdamWSchedule** — optimizer/schedule, eğitim döngüsüne özel. sequexa-core'a taşınacak.
- **CausalMask** — autoregressive attention maskesi, transformer'a özel. sequexa-core'a taşınacak.
- **MatMulWeightBwd** — ayrı bir primitive değil, belirli bir transpoz kombinasyonuyla matmul; `MatMulTrp` zaten kapsıyor, ayrı builtin olarak kalmıyor.
