# wilupgu Shader Refactor Planı

## Neden

wilupgu şu an 3 ayrı yerde her yeni kernel eklendiğinde şişiyor:

1. `src/backend.rs::kernel_layout` — TÜM kernellerin `TensorMode` binding şemasını tutan tek dev `match`.
2. `src/backends/wgpu.rs::kernel_src` — TÜM WGSL kaynaklarına giden `match` (`include_str!` ile).
3. `src/backends/cpu.rs::execute` ve `src/backends/cuda.rs::execute` — kernel ismi → Rust fonksiyonu/launch eşlemesi yapan `match` blokları.

Ember (NNUE) gibi wilupgu'yu kullanan her yeni proje, kendi kernellerini (BiasAdd, ClippedReLU, MseLoss gibi) eklemek için bu 4 dosyaya dokunmak zorunda kalıyor. Bu sürdürülebilir değil — wilupgu sonsuza kadar her tüketici projenin kernellerini içinde taşıyamaz.

Ayrıca CUDA kernel kaynakları şu an `src/nn/cuda_kernels.rs` içinde çirkin ham string sabitleri olarak duruyor — bu da ayrıca temizlenecek.

`nn/shaders.rs`'teki `BuiltInShader` enum'u kodda hiçbir yerde kullanılmıyor (grep ile doğrulandı) — tamamen kozmetik/ölü kod, refactor'de silinecek.

## Hedef

Her proje (akasha-core, ember, gelecekte başkaları) kendi `src/shaders/` dizinine sahip olsun, kendi kernellerini kendi crate'inde tanımlasın. wilupgu sadece:
- Gerçekten evrensel, yapısal kernelleri (MatMul ailesi, ResidualAdd, ZeroTensor, AdamW gibi) built-in olarak barındırsın.
- Generic bir "shader tanımlama" mekanizması sunsun ki tüketici projeler kendi kernellerini wilupgu'ya hiç dokunmadan tanımlayabilsin.

## Denenmiş ve elenen tasarım: string-keyed registry

İlk önerim: `Backend` trait'ine `type KernelImpl` associated type + `register_kernel(name, layout, impl)` metodu eklemek, her backend'in kendi `Mutex<HashMap<String, ...>>` registry'sini tutması, built-in'lerin `::new()` içinde kendini register etmesi.

**Neden elendi**: Bu tasarımda downstream proje her backend'i (cpu/wgpu/cuda) ayrı ayrı, backend kurulduktan hemen sonra register etmek zorunda kalıyor. Farklı backend'ler farklı cargo feature'ların (`cpu`, `cuda`) arkasında opsiyonel olduğu için, bu register çağrıları `#[cfg(feature = "cuda")]` gibi bayraklarla sarılmak zorunda kalıyor. Sonuç: "bu kernel hangi backend'lerde var" sorusunun cevabı kodda dağınık cfg bloklarını gezerek bulunuyor, tek bakışta görünmüyor. Kullanıcı (Hüseyin) bunu haklı olarak reddetti.

## Kabul edilen tasarım: statik `Shader` struct'ı

Kernel kimliği `&str` değil, `&'static Shader` referansı olacak. Bir kernelin TÜM backend implementasyonları (veya kasıtlı yokluğu) TEK bir struct literal'inde toplanıyor — mutable registry yok, `register_kernel` çağrısı yok, cfg flag'i gezmeye gerek yok.

```rust
// wilupgu/src/shader.rs — feature-gate'siz, her zaman derlenir

pub struct Shader {
    pub name: &'static str,           // debug label + pipeline-cache/panic mesajları için
    pub layout: &'static [TensorMode],
    pub wgpu: Option<&'static str>,   // WGSL kaynağı
    pub cpu: Option<fn(&[CpuBinding])>,
    pub cuda: Option<CudaSpec>,
}

pub struct CudaSpec {
    pub src: &'static str,
    pub entry: &'static str,
    pub shape: CudaShape,
}

pub enum CudaShape {
    InOut1,                                    // 1 InOut buffer (+ opsiyonel meta)
    In2Out1,                                    // 2 Input + 1 Output
    Add,                                        // elementwise accumulate
    Custom(fn(&CudaBackend, &[CudaBinding])),   // MatMul/Embedding gibi yapısal olanlar için kaçış kapısı
}
```

### Neden CUDA'da düz `Option<&'static str>` yetmiyor

WGPU zaten generic: workgroup sayısını caller (`add_node`) veriyor, shader kendi indexliyor, `build_node` sadece WGSL'i derleyip bind group kuruyor. CPU zaten generic: fn pointer alıyor, marshaling yok. Ama CUDA'da `cudarc`'ın `launch_builder().arg(...)` API'si her kernelin parametre sırasına göre elle doldurulmalı. Bugün zaten var olan `launch_inout_1`/`launch_in2_out1`/`launch_add` gibi generic launcher fonksiyonları bu ortak "şekilleri" kapsıyor — yeni elementwise kernel eklerken (BiasAdd, ClippedReLU, MseLoss gibi) bu şekillerden birine uyduğun için yeni Rust glue yazmana gerek kalmıyor, sadece `shape: CudaShape::InOut1` gibi bir seçim yapıyorsun. Yapısal kernel'ler (matmul, embedding, attention benzeri) için `Custom(fn)` kaçış kapısı var — bunlar zaten wilupgu built-in kalacak kernel'ler.

### cfg sızıntısını önleme

`Shader`/`CudaSpec`/`CpuBinding`/`CudaBinding` gibi payload tipleri düz veri (string + fn pointer) — `cudarc` veya gerçek GPU makinesine bağımlı değiller. Bu yüzden wilupgu'nun **feature-gate'siz çekirdeğinde** tanımlanacaklar. Sadece gerçek `CudaBackend` (asıl `cudarc` kullanan, ağır makine) `cuda` feature'ının arkasında kalacak. Böylece downstream'de bir `Shader` literal'i yazarken `cuda` alanına `Some(spec)` ya da `None` yazmak için hiçbir zaman `#[cfg(...)]` gerekmiyor — struct her zaman derlenir.

`CpuBinding` ve `CudaBinding` tipleri şu an muhtemelen `pub(crate)` — bunları `pub` yapmak gerekecek ki downstream `fn(&[CpuBinding])` yazabilsin.

### Pipeline cache key: isimden pointer'a

WGPU'nun `pipeline_cache: HashMap<String, ...>` bugün kernel ismini string olarak key yapıyor. İki farklı proje (ember, akasha) yanlışlıkla aynı ismi seçerse (`"Add"` gibi) teorik çakışma riski var. `&'static Shader`'ın pointer'ını (`shader as *const Shader as usize`) key yaparsak bu risk tamamen ortadan kalkar. Aynı mantık CUDA'nın kendi derleme cache'i için de geçerli.

## Değişecek dosyalar ve değişimin boyutu

- `src/graph.rs::ComputeGraph::add_node` — parametre `kernel: &str` yerine `shader: &'static Shader` olacak. `kernel_layout(kernel)` çağrısı yerine `shader.layout` doğrudan okunacak. Küçük değişiklik.
- `src/backend.rs::kernel_layout` — **tamamen silinecek**. `Backend::build_node` trait metodunun imzası `kernel: &str` yerine `shader: &'static Shader` alacak şekilde güncellenecek.
- `src/backends/wgpu.rs` — `kernel_src` fonksiyonu **tamamen silinecek**. `build_node` içinde `shader.wgpu.unwrap_or_else(|| panic!("[wgpu] shader `{}` has no wgpu impl", shader.name))` satırına inecek. Pipeline cache key'i pointer'a çevrilecek.
- `src/backends/cpu.rs` — `execute` içindeki ~30 satırlık `match` **tamamen silinecek**. `build_node` (veya execute) `shader.cpu.unwrap_or_else(...)` ile fn pointer'ı `CpuNode` içine gömecek; `execute` sadece `for node in nodes { (node.func)(&node.bindings) }` olacak.
- `src/backends/cuda.rs` — `execute` içindeki büyük `match`, `CudaShape` enum'una göre 4 kola inecek (`InOut1 => launch_inout_1(...)`, `In2Out1 => ...`, `Add => ...`, `Custom(f) => f(self, bindings)`). Mevcut `launch_*` fonksiyonlarının çoğu (matmul, embedding, rmsnorm gibi yapısal olanlar) `Custom` fn pointer'ları olarak built-in `Shader` sabitlerine gömülecek, silinmeyecek.
- `src/nn/shaders.rs` — `BuiltInShader` enum'u **tamamen silinecek** (zaten kullanılmıyordu).
- `src/nn/cuda_kernels.rs` — içindeki ham CUDA string sabitleri built-in `Shader` sabitlerinin `cuda.src` alanlarına taşınacak; dosya muhtemelen `builtin.rs` gibi bir isimle, `Shader` sabitleriyle birlikte yeniden düzenlenecek.
- `src/shaders/*.wgsl` (mevcut tüm dosyalar) — sadece wilupgu built-in kalacak kernellerin (MatMul ailesi, ResidualAdd, ZeroTensor, AdamW) WGSL'leri wilupgu'da kalacak. Geri kalanlar (SiLU, RoPE, RMSNorm, CausalSoftmax, CrossEntropy, Embedding tartışmalı, BiasAdd/ClippedReLU/MseLoss) ilgili projenin kendi `src/shaders/`'ına taşınacak.

## Built-in kalacak kernel kapsamı (üzerinde anlaşıldı)

Gerçekten her sinir ağının ihtiyaç duyacağı, dimension-agnostic, performans-kritik olanlar:
- `MatMul`, `MatMulTrp`, `MatMulAdd`, `MatMulWeightBwd` (matris çarpımı ailesi — en sık çağrılan, elle optimize edilmeyi hak eden)
- `ResidualAdd`, `BwdAddInplace`
- `ZeroTensor`
- `AdamW`

Bunların dışındaki her şey (SiLU, RoPE, RMSNorm, CausalSoftmax, CrossEntropy, Embedding, BiasAdd, ClippedReLU, MseLoss ve bunların backward'ları) ilgili projeye (akasha-core veya ember) taşınacak. `Embedding`/`EmbeddingBwd` sınırda ama muhtemelen akasha-core'a taşınacak (NNUE de kullanıyor ama proje-agnostik sayılmayacak kadar spesifik).

Built-in'ler artık enum değil, wilupgu içinde tanımlı `pub static` `Shader` sabitleri olacak (örn. `wilupgu::builtin::MATMUL`).

## Adım adım plan

1. `src/shader.rs` dosyasını oluştur: `Shader`, `CudaSpec`, `CudaShape` tiplerini tanımla. `CpuBinding`/`CudaBinding` tiplerini `pub` yap.
2. `Backend` trait'inin `build_node` imzasını `shader: &'static Shader` alacak şekilde güncelle.
3. `ComputeGraph::add_node`'u güncelle, `kernel_layout` fonksiyonunu sil.
4. CPU backend: `execute`'daki match'i sil, fn-pointer bazlı dispatch'e geç.
5. WGPU backend: `kernel_src` fonksiyonunu sil, pipeline cache'i pointer-keyed yap.
6. CUDA backend: `execute`'daki match'i `CudaShape` enum'una göre 4 kola indir, mevcut `launch_*` fonksiyonlarını `Custom` fn pointer'ı olarak koru.
7. wilupgu içinde built-in `Shader` sabitlerini tanımla (MatMul ailesi, ResidualAdd, ZeroTensor, AdamW) — WGSL/CUDA kaynakları mevcut `.wgsl`/`cuda_kernels.rs` içeriğinden taşınacak.
8. `nn/shaders.rs`'teki `BuiltInShader` enum'unu sil.
9. akasha-core'da `src/shaders/` dizinini aç, built-in olmayan kernellerin WGSL/CPU-fn/CUDA kaynaklarını oraya taşı, kendi `Shader` sabitlerini tanımla, `add_node` çağrılarını `&'static Shader` alacak şekilde güncelle.
10. ember'de aynısını yap: `BiasAdd`, `ClippedReLU`, `MseLoss` + backward'ları kendi `src/shaders/`'ına taşı, `Shader` sabitlerini tanımla.
11. `cargo test --features cpu` (wilupgu) + akasha-core `cargo check --lib` + ember `cargo test` ile regresyon kontrolü.

## Tahmini süre

- Adım 1-8 (wilupgu çekirdek refactor): 3-4 saat.
- Adım 9 (akasha-core taşıma): 1-2 saat.
- Adım 10 (ember taşıma): NNUE zaten az kernel içerdiği için ~30-45 dakika.

## Bu refactor'den SONRA sırada bekleyen işler (bu oturumda konuşuldu, henüz başlanmadı)

Kabul edilen sıralama:
1. **Bu shader refactor'ü** (yukarıdaki plan).
2. Ember'in yeni kernellerini (BiasAdd/ClippedReLU/MseLoss + bwd) CUDA'ya da taşımak — yeni `Shader` mekanizması üzerinden, eski usul `cuda_kernels.rs`'e gömerek DEĞİL. Öncelik CUDA çünkü gerçek eğitim arkadaşın NVIDIA GPU'sunda yapılacak (kullanıcının iGPU'su pratik değil).
3. Ember için `ModelConfig` struct'ı (num_features, hidden, l1, l2, batch_size, k_slots vb. — akasha-core'daki gibi tek yerde toplanacak, "tek seferde" bitirilecek) + **iki perspektifli (dual-perspective) NNUE mimarisi**: tek accumulator yerine white/black king bakış açısından iki `FeatureTransformer` (aynı `table` ağırlığını paylaşan), L1 ağırlık matrisi iki yarıya bölünüp (`W_white`, `W_black`) her perspektifin `activated` çıktısı kendi yarısıyla ayrı matmul'lenip toplanacak (fiziksel concat kernel'i gerekmiyor).
4. Quantization: wilupgu'ya **FakeQuantize** kernel'i eklenecek (forward: `round(clamp(x,-1,1)*scale)/scale` ile int8 hassasiyet kaybını simüle eder; backward: straight-through estimator, gradyan olduğu gibi geçer). Hem akasha-core hem ember faydalanacak, "built-in" kernel listesine girecek. Gerçek int8 paketleme (export sırasında) wilupgu kernel'i değil, düz host-side Rust serileştirme olacak (mevcut `export_bin` fonksiyonuna eklenecek).
5. Akasha-core'a Flash Attention eklenmesi (çoğunlukla shader eklentisi, akasha-core'un kendi kodunda az değişiklik).

## Ertelenen (bugün/yarın değil, sonraya kalan) işler

- Ignis (C++ satranç motoru) entegrasyonu.
- FEN → feature-index dataset loader (Ember için).

Bu ikisi kullanıcının açık talimatıyla erteleniyor: "ignis kısmını bugün bile değil yarın yaparız. o yüzden dataset loarder sonraya kalabilir."

## Ek bölüm: f16/bf16 altyapısı + launch_* boilerplate temizliği (2026-07-06)

Yukarıdaki plan tamamlandıktan sonraki oturumda (FlashAttention, kernel
fusion, CUDA TF32/Graphs, AdamW on-device schedule bittikten sonra) ortaya
çıkan yeni, ayrı bir refactor turu. Aşağıdaki maddeler sırayla uygulanıyor.

### 1. [DONE] `launch_*` boilerplate -> `define_launch!` makrosu

**Sorun:** `backends/cuda.rs`'te ~25 fonksiyon, hepsi aynı iskelet
(meta byte'larını parse et -> buffer'ları kilitle -> grid config hesapla ->
`launch!` çağır), sadece alan sayısı/mutability/grid formülü farklı.
Sadece imzalar bile onlarca satır.

**Neden generic değil, makro:** Fonksiyonlar arasındaki fark tip parametresi
değil, *yapısal* (kaç buffer, hangisi mut, kaç meta alanı, grid formülü) --
bu, Rust'ta generic'lerin değil makroların çözdüğü bir tekrar türü.

**Tasarım:** İki makro:
- `read_meta!(bytes, a: u32, b: f32, ...)` -- ardışık byte-offset parse'ı
  tek satıra indirir.
- `define_launch!(name, meta_slot: N, meta: [...], buffers: [mut/ro isim: slot, ...], let: [...], grid: expr, launch: [args])`
  -- tüm fonksiyon gövdesini üretir. `meta` ve `let` blokları opsiyonel
  (`$(...)?`), bazı kernel'lerde meta yok (`n` buffer uzunluğundan geliyor).

**Doğrulama:** Bu makine CUDA'sız olduğu için `backends/cuda.rs` derlenemiyor
(cudarc'ın `build.rs`'i nvcc arıyor). Makronun kendisini (hijyen -- meta'dan
okunan değişkenlerin `grid`/`launch` ifadelerinde görünür olması, karışık
mut/ro buffer kilitleme, karışık u32/f32 meta) gerçek CudaSlice yerine sahte
`Mutex<Vec<f32>>` ile izole bir harness'ta `rustc --edition 2021` ile fiilen
derleyip çalıştırarak doğruladım (5 farklı shape, hepsi doğru grid/değer
üretti). Gerçek dosyadaki nihai derleme testi yine de arkadaşının CUDA'lı
makinesinde olacak.

**Kapsam dışı bırakılanlar (bilinçli, hand-written kalıyor):**
- `gemm_matmul`, `gemm_weight_bwd` -- cuBLAS çağrısı, `launch!` kalıbına
  girmiyor (madde 4'te generic-over-T olacak).
- `launch_adamw` -- iki ayrı meta okuması (slot 4 + slot 6) artı canlı
  `schedule` buffer'ı; bu akşam öncesi yeni doğrulanmış kodu riske atmamak
  için dokunulmadı.
- `launch_embedding` -- diğerlerinden farklı olarak `wg: [u32;3]` parametresi
  alıyor (tek istisna, imza şekli uyuşmuyor).

Geri kalan ~23 fonksiyon `define_launch!` çağrısına indirildi, isim/imza/
davranış birebir aynı kaldı (akasha-core'daki çağıranlar dokunulmadı).
`launch_causal_mask`/`launch_adamw_schedule` istisnaen imza değiştirdi
(sabit `src`/`func`'ları artık parametre olarak alıyorlar, tekdüzelik için)
-- bunların tek çağıranı aynı dosyadaki `custom_causal_mask`/
`custom_adamw_schedule`, onlar da güncellendi.

### 2. [DONE] `Dtype` enum + `Backend` trait'e zorunlu dtype-parametreli metodlar

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype { F32, F16, Bf16 }
```
`Backend` trait'ine (default impl YOK, her backend kendi impl'ında açıkça
yazacak):
```rust
fn alloc_dtype(&self, elem_count: usize, dtype: Dtype) -> Self::Buffer;
fn upload_as(&self, buf: &Self::Buffer, data: &[f32], dtype: Dtype);
fn download_as(&self, buf: &Self::Buffer, dtype: Dtype) -> Vec<f32>;
```
(`download_as_f32` değil `download_as` -- quantizasyon çalışmasında dönüş
tipi de genişleyebilir, o zaman tekrar ele alınacak.)
CPU/wgpu bugünkü haliyle sadece `Dtype::F32` dalını yazar (açık, görünür
karar -- sessiz miras alınan default değil). CUDA gerçek f16/bf16
dönüşümünü (`half::f16::from_f32`/`to_f32`, bit-reinterpret değil, gerçek
yuvarlama) implement eder.

**Doğrulandı:** `cargo check --features cpu` (wgpu+cpu backend'lerini
kapsıyor, cuda hiç dokunmuyor) temiz geçti -- yeni trait metodları her iki
backend'de de gerçekten derlendi. akasha-core `cargo check --lib --features
vulkan` de temiz (3 önceden bilinen dead-code uyarısı dışında, onlar bu
değişiklikle ilgisiz).

### 3. [DONE] `CudaBuffer` -> enum

```rust
#[derive(Clone)]
pub enum CudaBuffer {
    F32(Arc<Mutex<CudaSlice<f32>>>),
    F16(Arc<Mutex<CudaSlice<half::f16>>>),
    Bf16(Arc<Mutex<CudaSlice<half::bf16>>>),
}
```
Simetrik, eşit ağırlıklı erişim: `lock_f32(bindings, slot)` /
`lock_f16(bindings, slot)` serbest fonksiyonları (ne biri "asıl yol" ne
diğeri "özel durum" -- ikisi de `find`/`meta_bytes` gibi kardeş yardımcılar).
Mevcut ~23 `define_launch!` çağrısı `buffers: [mut g: 0]` gibi yazıldığı
için, `@lock` kolunun içindeki `.slice.lock().unwrap()` -> `.slice.as_f32().lock().unwrap()`
değişimi TEK YERDE (makronun `@lock` kolunda) yapıldı -- çağıran ~23
satırın hiçbiri değişmedi. Bu, makronun bu enum geçişini bile
kolaylaştırdığının kanıtı.

**Gerçek bug riski (yakalanmış VE düzeltilmiş):** `BufferPool<Buf>` sadece
`size_bytes: u64` ile anahtarlanıyordu -- F16 alloc'u aynı byte sayısında
bir F32 buffer'ı geri alabilirdi. `pool.rs`'te `BufferPool<Buf, K=u64>`
generic key parametresi eklendi, CUDA tarafı `BufferPool<CudaBuffer,
(u64, Dtype)>` kullanıyor artık; wgpu/cpu tarafı `K=u64` default'uyla hiç
değişmeden kaldı (`cargo check --features cpu` ile doğrulandı).

`CudaBackend::alloc_dtype`/`upload_as`/`download_as` de yazıldı (gerçek
`half::f16::from_f32`/`to_f32` dönüşümüyle, bit-reinterpret değil).
`alloc`/`alloc_from_cpu`/`copy_from_cpu`/`copy_to_cpu` (eski, F32-sabit
API) ve `gemm_matmul`/`gemm_weight_bwd`/`launch_adamw`/`launch_embedding`/
`build_node`'daki tüm doğrudan `.slice.lock()` çağrıları `.as_f32()`
üzerinden geçecek şekilde güncellendi -- bu makine CUDA'sız olduğu için
`cargo check --features cuda` çalışmıyor (build.rs nvcc arıyor, öncekiyle
aynı, yeni bir hata değil), asıl derleme testi yine arkadaşının makinesinde
olacak. `half`/`cudarc "f16"` feature'ları Cargo.toml'a eklendi; cudarc'ın
vendored kaynağından `DeviceRepr`/`ValidAsZeroBits` impl'lerinin `half::f16`/
`half::bf16` için `#[cfg(feature="f16")]` arkasında gerçekten var olduğu
doğrulandı (tahmin değil, kaynağı okudum).

### 4. [PLANNED] GEMM generic-over-T (gerçek Rust generic, makro değil)

cudarc'ın kendi `Gemm<T>` trait'i f32/f16/bf16 için hazır, çağrı şekli
üçünde de aynı:
```rust
fn gemm_matmul_generic<T>(&self, a: &CudaSlice<T>, b: &CudaSlice<T>, c: &mut CudaSlice<T>, transpose_b: bool, beta: f32)
where CudaBlas: Gemm<T>
```
`gemm_matmul`/`gemm_matmul_f16`/`gemm_matmul_bf16` bu tek çekirdeğe ince
sarmalayıcılar olacak -- bf16 için "yeni launch fonksiyonu" değil, aynı
gövdeye üçüncü ince kapı.

### 5. [PLANNED] CUDA-C kernel kaynağına f16 template'i (string-swap, güvenli hali)

Kör `.replace("float","half")` YAPILMAYACAK (RMSNorm/softmax/cross-entropy
gibi reduction yapan kernellerde toplama hassasiyetini bozar). Bunun yerine
her kernel kaynağının başına iki typedef:
```c
typedef float scalar_t; // depolama tipi -- f16 varyantında __half olur
typedef float acc_t;    // reduction/toplama tipi -- HER ZAMAN float kalır
```
f16 varyantı üretmek = sadece `scalar_t` typedef satırını değiştirmek.
Bugün sadece GEMM'e uygulanacak (madde 4), elementwise kernellerin f16
versiyonu şimdilik gereksiz (bkz. madde 6).

### 6. Kapsam notu

f16/bf16'nın bugün gerçekten ihtiyacı olan tek yer **matmul** (bant
genişliği + tensor-core kazancı esas orada). RMSNorm/RoPE/softmax gibi
~20 elementwise kernel şimdilik f32-only kalıyor, dokunulmuyor.
bf16, WGSL'de native tip olmadığı için CUDA-only; wgpu tarafı zaten
dtype-agnostic (`WgpuBuffer = Arc<wgpu::Buffer>`, ham byte) olduğu için
mimari değişiklik gerektirmiyor, ileride sadece yeni WGSL shader + f16
`alias` + `Features::SHADER_F16` isteği gerekecek.

Akasha-core'un kendi shader'larını (hepsi `array<f32>`) f16'ya geçirmek
ayrı, ileride ele alınacak bir iş -- bugünkü kapsam sadece wilupgu.

## Notlar / hatırlatmalar

- `backend_parity` testindeki segfault, `git stash`/`git stash pop` ile pristine (değişikliklerden önceki) kodda da aynı şekilde reprodüklendiği için **wilupgu'nun yeni eklemelerinden kaynaklanmadığı kanıtlandı** — muhtemelen bu sandbox ortamında GPU adapter eksikliği/uyumsuzluğu. Kullanıcının gerçek makinesinde (CUDA'lı) muhtemelen sorun çıkarmaz.
- akasha-core "sacred" proje — tüm wilupgu değişiklikleri additive/geriye uyumlu olmalı, her adımdan sonra `cargo check --lib` ile doğrulanmalı.
- Ember'in mevcut smoke test'i (`ember/tests/smoke.rs`, `training_step_reduces_loss`) CPU backend üzerinde uçtan uca çalışıp loss'un düştüğünü doğruluyor — refactor sonrası bu testin hâlâ geçtiğinden emin olunmalı.
