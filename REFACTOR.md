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

## Notlar / hatırlatmalar

- `backend_parity` testindeki segfault, `git stash`/`git stash pop` ile pristine (değişikliklerden önceki) kodda da aynı şekilde reprodüklendiği için **wilupgu'nun yeni eklemelerinden kaynaklanmadığı kanıtlandı** — muhtemelen bu sandbox ortamında GPU adapter eksikliği/uyumsuzluğu. Kullanıcının gerçek makinesinde (CUDA'lı) muhtemelen sorun çıkarmaz.
- akasha-core "sacred" proje — tüm wilupgu değişiklikleri additive/geriye uyumlu olmalı, her adımdan sonra `cargo check --lib` ile doğrulanmalı.
- Ember'in mevcut smoke test'i (`ember/tests/smoke.rs`, `training_step_reduces_loss`) CPU backend üzerinde uçtan uca çalışıp loss'un düştüğünü doğruluyor — refactor sonrası bu testin hâlâ geçtiğinden emin olunmalı.
