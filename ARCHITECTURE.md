# wilupgu — Mimari

## Ne bu?

Genel tanıtım [README.md](README.md)'de. Bu dosya iç mimariyi anlatır: katman
haritası, çekirdek kontratlar, yaşam döngüleri, backend farkları ve kernel
ekleme checklist'i.

## Katman haritası

```
callers: akasha-core, ember
   │  (Shader statiği + Binding'ler + grid ile node ekler)
   ▼
┌─ Shader ('static) ─────────────────────────────────────────────┐
│ name + layout: &[TensorMode] + wgsl / cpu / cuda kaynakları    │
└────────────────────────────────────────────────────────────────┘
   │  kod değil VERİ: builtin/ 12 tanesini sağlar, caller kendi
   │  statiklerini ekleyebilir (akasha-core shaders/ bunu yapar)
   ▼
┌─ ComputeGraph<B> ──────────────────────────────────────────────┐
│ add_node(shader, bindings, grid)                               │
│   → Binding.mode ↔ Shader.layout assert'i burada              │
│ execute() / execute_captured(id)                               │
└────────────────────────────────────────────────────────────────┘
   │  build_node ile backend'e özgü B::Node üretir ve saklar
   ▼
┌─ Backend trait ────────────────────────────────────────────────┐
│ alloc / recycle (pool)   copy_from/to_cpu   build_node         │
│ execute / execute_captured / synchronize                       │
└────────────────────────────────────────────────────────────────┘
   │                     │                      │
   ▼                     ▼                      ▼
WgpuBackend          CudaBackend            CpuBackend
wgsl → pipeline      NVRTC + cuBLAS         fn(&[CpuBinding])
pow2 pool            exact-size pool        exact-size pool
in-flight ≤ 16       graph capture          senkron düz döngü
                     (Warmed → Captured)

dikey kesit (her katmanın yanından geçer):
  Tensor<B> = ctx + B::Buffer + mantıksal boyut.
  Drop'ta sole-owner ise buffer pool'a döner; Node'lar buffer'lara
  Arc tuttuğu için canlı bir graph'ın buffer'ı asla erken pool'a düşmez.
```

Katman başına tek sorumluluk:

- **Shader**: bir kernel'in üç backend'deki kimliği. Davranış içermez, sadece kaynak + kontrat (layout) taşır.
- **ComputeGraph**: node listesi + kontrat doğrulama. Backend'den habersizdir; sıralamayı caller'ın verdiği ekleme sırası belirler.
- **Backend**: belleğin ve çalıştırmanın tek sahibi. Pool politikası, meta okuma şekli ve capture semantiği backend'e özgüdür (asimetriler için aşağıdaki tabloya bak).
- **Tensor**: buffer'ın yaşam döngüsü sahibi; mantıksal boyutu bilen tek yer.

## Çekirdek kontratlar

Kodun kendisinin söyleyemediği kurallar. Her madde bir garanti, yükümlülük
veya tuzaktır; ihlalini derleyici değil bu liste yakalar. Yeni kural
keşfedildikçe buraya satır eklenir.

- **TensorMode = davranış beyanı, süs değil.** `Output`: kernel HER elemanı
  yazmak zorundadır — pool çöpü dahil ezilir, caller init'e güvenmez.
  `Accumulate`: kernel `+=` yapar, init CALLER'ın yükümlülüğüdür. `InOut`:
  eski değer okunup tüketilir. `Meta`: read-only bağlanır (wgpu'da fiziken
  read-only storage — kernel metaya yazamaz). Etiket kernel'in gerçek
  davranışıyla eşleşmek zorunda; yalan etiket sessiz çöp-grad üretir
  (T1 sınıfı bug).
- **add_node her binding'i layout'a karşı assert eder** (mode birebir eşit,
  slot menzil içinde). Tek kapı budur; backend'ler ayrıca doğrulamaz.
- **Aynı buffer bir dispatch'te iki slota bağlanamaz** — CUDA'da assert
  (çift-lock deadlock önlemi, T2); kavramsal olarak her backend'de yasak.
- **Meta okuma zamanı backend'e göre değişir** (Backend farkları tablosu).
  Türetilmiş kural: matmul içeren graph'ın metası değişecekse o graph
  `execute_captured` KULLANAMAZ — cuBLAS boyutları capture'da donar.
- **Her şey f32 word olarak saklanır**: alloc %4 katı (CUDA assert'li),
  `alloc_from_cpu`'da `T` 4-byte'ın katı olmalı. F16/BF16 yalnız CUDA
  `alloc_dtype` yolunda yaşar.
- **Fiziksel ≠ mantıksal boyut** (wgpu pow2 class): `copy_to_cpu` HAM fiziksel
  içeriği döndürür, kuyruğu çöptür — dışarıdan hep `Tensor::to_cpu`
  (truncate eden) kullanılır. `arrayLength` fiziksel sınırdır; mantıksal
  sınırı meta/grid sağlar.
- **Zero-init garantisi yoktur** — pool'dan dönen buffer çöp içerir. Temiz
  içerik isteyen `init_from_cpu` ya da ZERO_TENSOR kullanır.
- **wgpu dispatch ekseni ≤ 65535 workgroup.** 16.7M+ elemanlı tensöre düz 1D
  grid atan kernel sessizce eksik çalışır ya da device kaybettirir —
  2D-linearize desen şart (zero_tensor / adamw / grad_scale örnekleri).
- **dispatch_generic kısıtları** (CUDA): en fazla BİR Meta slot ve Meta SON
  slot olmalı; generic yol yalnız F32 buffer kilitler. Aşan kernel
  `CudaShape::Custom` yazar.
- **`execute_captured`'dan sonra graph'a node ekleme.** CUDA eski kaydı
  oynatır, wgpu güncel listeyi koşar — iki backend sessizce ayrışır.
  Graph'i kur, bitir, sonra çalıştır.
- **Sıralama tek dayanaktır**: tek queue/stream işleri sırayla yürütür;
  `copy_from_cpu` senkronsuz güvenlidir ve pool reuse güvenliği tamamen buna
  dayanır (bkz. Buffer yaşam döngüsü uyarısı).

## Buffer yaşam döngüsü

```
alloc(size) ─ pool'da uygun class var mı? ─ evet → recycled buffer (İÇERİK ÇÖP)
     │ hayır
     ▼
driver'dan yeni buffer
     │
kullanım — Tensor + onu bind eden Node'lar Arc'ı paylaşır
     │
Tensor::drop ─ sole owner mı? ─ hayır → hiçbir şey (Node yaşatmaya devam eder)
     │ evet
     ▼
pool.recycle ─ bucket dolu mu (8)? ─ evet → destroy
     │ hayır
     ▼
free-list'te bekler → bir sonraki alloc'ta geri döner
```

- Recycled buffer'ın içeriği **çöptür** — panzehiri Output kontratı (tam ezme)
  veya `init_from_cpu`. Sıfır-init'e güvenen kod yazılamaz.
- Node'lar binding buffer'larına Arc tutar → canlı bir graph'ın buffer'ı sole
  owner olamaz, erken pool'a düşemez. Graph drop olunca buffer'lar da serbest
  kalır (Tensor hâlâ yaşıyorsa onda, o da düştüyse pool'da).
- **Neden GPU-güvenli**: tek queue (wgpu) / tek stream (CUDA) işleri sırayla
  yürütür; recycle edilmiş buffer'ı yeniden kullanan iş, onu bırakan işten
  sonra koşar. **UYARI: çoklu-stream/queue eklenirse bu varsayım çöker** —
  o gün pool'a fence/event senkronu gerekir, yoksa use-after-free sınıfı doğar.
- Fiziksel ≠ mantıksal boyut (yalnız wgpu, pow2 class): `Tensor.size`
  mantıksaldır, `to_cpu` ona truncate eder; buffer kuyruğu çöptür ve
  `arrayLength` guard'ları FİZİKSEL boyuta karşı çalışır — mantıksal sınır
  denetimi değildirler, onu meta/grid sağlar.
- `free_buffer` pool'u atlar, anında destroy — istisnai yol.

## Execute yolları

`execute()` — node listesini backend'e verir:

- **wgpu**: tüm node'lar tek command encoder + tek compute pass'e dizilir,
  tek submit. Submission index'i in-flight kuyruğuna girer; kuyruk
  MAX_IN_FLIGHT'ı (16) aşarsa yalnız EN ESKİ submission beklenir
  (`WaitForSubmissionIndex`), aşmazsa sadece `Poll`. Amaç: GPU'yu aç
  bırakmadan sınırsız kuyruklanmayı da önlemek (H7).
- **CUDA**: node'lar sırayla stream'e dispatch edilir; backpressure yok,
  stream sıralaması yeter.
- **CPU**: fn pointer'lar sırayla çağrılır, zaten senkron.

`execute_captured(key)` — default implementasyon düz `execute`'tur (wgpu/cpu
için fark yok). CUDA'da üç aşamalı yaşam döngüsü:

```
1. çağrı  → düz execute (warm-up: cuBLAS'ın ilk-çağrı alloc'ları
            capture DIŞINDA kalsın)                        → Warmed
2. çağrı  → stream capture ile kaydet → CudaGraph → launch → Captured
            (cuBLAS boyutları BU ANDA cached_meta'dan donar)
3+. çağrı → graph.launch() — dispatch maliyeti yok
```

- Kurallar: yalnız **sabit metalı** graph'ler captured kullanabilir (donma
  yüzünden); captured graph yakalandığı thread'den launch edilmelidir
  (debug_assert var — CudaGraph thread-safe değil).
- `ComputeGraph::drop` → `release_captured(id)`: yakalanan graph cache'ten
  düşer; her ComputeGraph'ın id'si benzersizdir (atomic sayaç).

## Backend farkları

|  | wgpu | cuda | cpu |
|---|---|---|---|
| Kernel kaynağı | WGSL (`Shader.wgpu`), pipeline cache (anahtar = Shader adresi) | CUDA C string → NVRTC → PTX, kernel cache | `fn(&[CpuBinding])`, doğrudan çağrı |
| Matmul yolu | kendi tiled WGSL'i / m=1'de GEMV | **cuBLAS** (TF32 açık) | naif üçlü döngü |
| Pool | **pow2 size class**, bucket başı 8, fazlası destroy | exact-size, anahtar (boyut, dtype) | exact-size |
| Fiziksel boyut | mantıksaldan BÜYÜK olabilir | == mantıksal | == mantıksal |
| Meta okuma | device'ta, execute anında (hep canlı) | generic: device pointer (canlı); cuBLAS: dispatch'te dtoh, capture'da donuk | host'ta, dispatch anında (canlı) |
| `execute_captured` | yok → düz execute | CUDA graph (Warmed → Captured) | yok → düz execute |
| Dtype | yalnız F32 (assert) | F32 / F16 / BF16 (gemm hepsinde; generic kerneller F32) | yalnız F32 (assert) |
| alloc %4 kuralı | assert YOK (bilinen asimetri — bug listesinde) | `size % 4 == 0` assert; her şey f32 word | assert yok |
| `synchronize` | `Maintain::Wait` + in-flight kuyruğu temizlenir | `stream.synchronize()` | no-op |
| Backpressure | in-flight ≤ 16, en eski beklenir | yok (stream sıralı) | anında |

## Yeni kernel ekleme checklist'i

Builtin ile downstream kernel arasında mekanizma farkı yoktur — ikisi de bir
`Shader` statiğidir. Yerleşim: builtin ise `src/builtin/` (+ cpu_kernels /
cuda_kernels), kendi projense kendi `shaders/` dizinin (akasha-core canlı
örnek). Sonrası ikisi için de aynı:

1. **Layout'u kernel DAVRANIŞINA göre yaz**, niyete göre değil: tam ezme →
   `Output`; `+=` → `Accumulate`; kısmi/oku-değiştir → `InOut`. Karar
   veremiyorsan kernel'in yazan satırına bak: `buf[i] = x` mi, `buf[i] += x` mi?
2. **Meta gerekiyorsa**: typed Pod struct; alan sırası WGSL/CUDA `struct Meta`
   ile byte-byte aynı; Meta SON slot; dispatch_generic başına en fazla bir Meta.
3. **WGSL**: `arrayLength` guard (fiziksel taşma) + meta/grid ile mantıksal
   sınır. `workgroup_size` ile grid formülünü BİRLİKTE tasarla; eleman sayısı
   16.7M'i aşabilecekse 2D-linearize grid.
4. **CUDA**: yetiyorsa `CudaShape::Generic` (block_dim + meta_fields);
   cuBLAS/özel akış gerekiyorsa `Custom`. `idx < n` guard'ını unutma.
5. **CPU**: düz Rust referansı yaz — parity testinin temelidir. Yazamıyorsan
   `cpu: None` + Shader statiğinin üstüne neden yorumu (boşluk bilinçli
   görünsün).
6. **Shared-memory reduction yazıyorsan**: `partial[0]` okunduktan sonra aynı
   diziye yazmadan önce `workgroupBarrier()` / `__syncthreads()`
   (rmsnorm_bwd dersi).
7. **Test**: CPU-referanslı doğrulama, grid sınırlarına DENK GELMEYEN
   boyutlarla (256/16 katı olmayan); fused kernel yazıyorsan unfused referansı
   `#[cfg(test)]` olarak yaşat.
8. **Son bakış**: Çekirdek kontratlar listesiyle çelişki var mı; downstream'sen
   projenin emitter kataloğuna satırını ekle.

## Test stratejisi

| Test | Neyi koruyor |
|---|---|
| `tests/backend_parity.rs` | builtin'ler CPU-referansına ve (cuda açıksa) birbirlerine karşı — matmul/trp, residual_add, causal_mask, zero_tensor |
| `tests/graph_chain_test.rs` | 10 zincirli matmul, satır başına ayrı değerle — eksik grid dispatch'i değer hatası olarak yakalar (T8) |
| `tests/advanced_graph_test.rs` | çok-node'lu graph + mode-mismatch paniği (idiot-proof testi) |
| `tests/cuda_meta_semantics.rs` | canlı meta / cached meta / capture-donması semantiği — **cfg(cuda): yalnız nvidia makinede derlenir ve koşar** |
| `src/backends/cuda.rs` (inline) | f16/bf16 gemm doğruluğu (cfg(cuda)) |

Kurallar:

- Her zaman `cargo test -- --test-threads=1` — paralel testler eşzamanlı
  GPU device'ları yüzünden segfault eder.
- Bu makinede nvcc yok: cuda feature hiç derlenmemiş kod içerebilir; CUDA'ya
  dokunan her değişiklik nvidia makinede bir tur ister.
- Yeni builtin = backend_parity'ye satır (checklist madde 7); GEMV/GEMV_ADD
  ve akasha'ya özgü kerneller akasha tarafındaki emit.rs testlerinde yaşar
  (bkz. akasha-core/ARCHITECTURE.md → Test haritası).
