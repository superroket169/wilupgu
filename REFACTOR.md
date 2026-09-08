# Bekleyen işler

Kapsam: sequexa-core + wilupgu + ember. Yalnız YAPILMAMIŞ maddeler; biten iş
buradan silinir (tarihçe git log'da). Sıra: doğruluk → hız → tasarım → feat.

## 🟠 Hız

**B9** — CUDA decode: her matmul dispatch'inde bloklayan dtoh (cuda.rs
gemm_meta_u32): capture dışında her cuBLAS çağrısı meta'yı device'tan senkron
çeker. Decode graph capture edilmiyor → token başına ~61 matmul × bloklayan
kopya. Decode'un matmul metaları sabit; matmul-family capture dışında da
cached_meta okusa maliyet kalkar. Dikkat: gerçekten dinamik meta'lı bir cuBLAS
çağrısı varsa ona opt-out gerekir. **Not:** sequexa ARCHITECTURE fikir
kuyruğundaki "decode'u cuBLAS'sızlaştırma" yapılırsa bu madde kökünden düşer —
önce onun kararı.

**B11** — Flash attention verimlilik notları (doğruluk tamam): (a)
thread-per-(row,head) tasarımı shared memory/tiling kullanmıyor; K/V her satır
için global'den tekrar okunuyor. (b) bwd_dkdv içteki döngüde d_i'yi her
(col,i) çifti için yeniden hesaplıyor — FlashAttention-2 gibi tek geçişte
D[i] = Σ dO·O precompute edilirse bwd'den koca bir head_dim döngüsü çıkar.

**B12** — Prefill her çağrıda graph'ı ve tüm ara buffer'ları yeniden kuruyor.
Pool yumuşatıyor ama prompt başına build + upload maliyeti var; uzunlukları
bucket'layıp graph cache'lemek mümkün.

## 🖥️ AMD/RADV ilk tam-ölçek koşusu (2026-08-18)

Yeni, kalıcı erişimli AMD test makinesi geldi (Ryzen 7 7700 + RX 7600 discrete
+ Raphael iGPU, Tailscale/SSH ile sınırsız erişim) — wgpu backend'inin
**production ölçeğinde (dim=768, vocab=50257) ilk gerçek koşusu** bu oldu
(laptop'ta `max_storage_buffer_binding_size` limiti yüzünden tam ölçek hiç
denenememişti; eğitim şimdiye kadar hep CUDA'da koştu). İki ayrı, bağımsız
bulgu çıktı:

**1) Flash attention register spill — ÇÖZÜLDÜ ve KAPATILDI (2026-08-30).**
`flash_attention.wgsl` / `flash_attention_bwd_dq.wgsl` /
`flash_attention_bwd_dkdv.wgsl` üçünde de `array<f32, MAX_HEAD_DIM>` tipinde
per-thread accumulator, döngü sınırı runtime Meta değeri (`m.head_dim`) olduğu
için derleyici tarafından unroll edilemiyor, register yerine VRAM-backed
scratch belleğe spill oluyordu (RADV hang dump'ında doğrulandı: VGPRs=16,
Scratch=64KB/wave) — hem ciddi yavaşlık hem de gerçek bir
`radv/amdgpu: GPU hang` sebebiydi. Fix: üç dosyada da sabit
`HEAD_DIM: u32 = 64u` (sequexa-hall'un tek konfigürasyonu). Bu, hardcode'un
head_dim≠64 için sessizce yanlış sonuç üretmesine açık kapı bırakmıştı —
kapatılan asıl kısım bu: sequexa-core `ops/emit.rs::assert_flash_head_dim`
artık her iki emitter'da (`flash_attention`, `flash_attention_bwd`)
`head_dim == 64` assert ediyor (tek-head_dim kabulü, shader üretimi değil —
sequexa tek bir modelin motoru, jenerik head_dim ihtiyacı yok). Bu assert
olmadan zaten 4 sequexa testi (gradcheck, batching, prefill, flash attention'ın
kendi testi) tiny config'lerde (head_dim=4/8/16) sessizce yanlış sayı
üretiyordu — testler artık hepsi head_dim=64'e taşındı, ayrıca guard'ın
gerçekten patladığını kanıtlayan bir `#[should_panic]` testi eklendi. CUDA
kernel'i hiç etkilenmedi (hâlâ runtime `head_dim` okuyor, `<=128` sınırıyla
genel). Detaylar: sequexa-core git log + wilupgu/SHADERS.md.

**2) wgpu otomatik senkronizasyon bug'ı — AÇIK, gerçek engel bu.**
Register-spill fix'i hang'i tam çözmedi: normal (async) çalıştırmada step
50 civarında loss bozuluyor (`0.0000`) ve kısa süre sonra
`Parent device is lost` ile çöküyor. `RADV_DEBUG=hang,syncshaders` VE
kendi eklediğimiz `WILUPGU_FORCE_SYNC=1` (wgpu.rs `execute()`'da her node'u
ayrı submit+wait ile çalıştıran deneysel branch, lokal, commit'lenmedi) —
**ikisi de bağımsız olarak** sorunu düzeltiyor: loss düzgün düşüyor, hang
olmuyor. Bu, wgpu'nun tek compute pass'e dizilen çok-node'lu graph'larda
(bizim fused train graph'ları gibi, 100+ node) dispatch'ler arası otomatik
bariyer eklemesinde gerçek bir eksiklik/gecikme olduğuna işaret ediyor —
wilupgu'nun kodu (backends/wgpu.rs `execute()`, tüm node'ları tek pass'e
diziyor) standart/beklenen wgpu kullanımı, elle bariyer eklemek API'de
mümkün değil. Şüpheli: **`wgpu = "0.19.4"`** (yayın: 2024-04-18) — proje
2026-06-22'de başladığında zaten 2+ yıl eskiydi (muhtemelen eski bir
tutorial/boilerplate'ten miras), güncel stable **30.0.0**'a kadar ~10 major
sürüm fark var. wgpu'nun "çok-kaynaklı compute pass'te bariyer" alanı
tarihsel olarak bilinen kırılgan bir köşe (gfx-rs/wgpu #5766, #2659, #6344,
PR #194, #3181) — upgrade kör bir bahis değil ama büyük bir iş (10 major
sürümlük API kırılması), bu oturumda başlanmadı.

**wgpu 0.19.4 → 30.0.1 upgrade YAPILDI (2026-09-08) — bariyer bug'ını ÇÖZMEDİ.**
Cargo.toml + backends/wgpu.rs, ~15 satırlık mekanik diff: `request_device`
tek argümana düştü (trace_path kalktı), `PipelineLayoutDescriptor`'da
`bind_group_layouts: &[Option<&_>]` + `push_constant_ranges`→`immediate_size`,
`ComputePipelineDescriptor`'a `compilation_options`/`cache` eklendi +
`entry_point` artık `Option<&str>`, `Device::poll` artık `Maintain` değil
`PollType` alıp `Result<PollStatus, PollError>` dönüyor, `get_mapped_range`
`Result` dönüyor, `ComputePass::set_bind_group` artık `Option<&BindGroup>`
istiyor. Derlendi; wilupgu 10/10 + sequexa-core 29/30 test gerçek HD620'de
geçti. **Ama aynı gün FORCE_SYNC'siz gerçek bir training run'ı hâlâ aynı
imzayla çöktü** (`Parent device is lost`, step ~4 civarı, resume sonrası) —
upgrade kod tabanını modernize etti ama bariyer davranışını değiştirmedi.
FORCE_SYNC hâlâ tek çalışan workaround.

**Sıradaki adımlar** (öncelik sırasıyla, hiçbiri bitmedi):
- Kısa vade: `WILUPGU_FORCE_SYNC=1` ile gerçek bir eğitim koşusu başlat
  (doğru ama yavaş — her node ayrı submit+wait, pipelining tamamen kayboluyor).
- Orta vade: tüm node'lar yerine sadece gerekli 1-2 sınırda senkron
  (bisection ile minimal bariyer noktasını bul, çoğu hızı geri kazan).
- Uzun vade (GÜNCEL PLAN, 2026-09-08 — wgpu upgrade'in yerini aldı): wgpu'yu
  bırakıp wilupgu'ya kendi native Vulkan backend'ini eklemek, bkz. aşağıdaki
  "🎨 Tasarım" bölümü. wgpu'nun opak/otomatik bariyerine bağımlı kalmak yerine
  node'ların kendi `Binding`/`TensorMode` bilgisinden gerçek hazard'ları
  çıkarıp elle bariyer basmak.

## 🎨 Tasarım: Büyük Wilupgu Refactoru (planlama, henüz başlanmadı — 2026-09-08)

Kapsam: backend trait'i ve backend setini yeniden düşünmek. Üç ayrı iş:

**1) Native Vulkan backend (`backends/vulkano.rs`, şimdilik boş dosya) —
wgpu'nun çözemediği bariyer bug'ını çözmek için.**
- Amaç: WGSL tek doğru shader kaynağı kalsın (ikinci bir shader dili YOK),
  ama dispatch'ler arası bariyer wgpu'nun otomatiğine değil kendi elimize
  bağlı olsun.
- Zincir: `naga::front::wgsl::parse_str` → `naga::valid::Validator` →
  `naga::back::spv::write_vec` (WGSL→SPIR-V — wgpu'nun içeride zaten yaptığı
  şey; `naga` wgpu'nun transitive dependency'si olarak elimizde) →
  `vulkano::shader::ShaderModule::new(device, ShaderModuleCreateInfo::new(&words))`
  (`unsafe`, doğrulamasız — ama naga zaten doğruladı, sorun değil). Pipeline +
  layout, wgpu.rs'teki `pipeline_cache` gibi shader başına bir kere kurulup
  cache'lenir.
- Bariyer: her node'un `Binding`/`TensorMode` listesi zaten hangi buffer'ı
  nasıl kullandığını taşıyor (Input/Output/InOut/Accumulate/Meta) — ardışık
  node'lar arası gerçek RAW/WAW hazard'larını buradan çıkarıp sadece gereken
  yere `vkCmdPipelineBarrier` basmak FORCE_SYNC'ten hızlı, wgpu'nun
  otomatiğinden doğru olmalı.
- **filuplex incelendi (2026-09-08, `~/Documents/Codes/filuplex`), arşivden
  çıkarılmadı — kod olarak değil, ders olarak faydalı:**
  - Tekrar kullanılabilir: `Context` (Instance/Device/Queue/allocator
    kurulumu) temiz ve minimal; ve önemlisi,
    `ShaderModule::new(device, ShaderModuleCreateInfo::new(words))` ile ham
    SPIR-V word'lerini runtime'da yüklemenin bu tam donanımda (HD620, vulkano
    0.35.1) **çalıştığı zaten kanıtlanmış** (`ops.rs::BuiltInShader::load_from_file`,
    `.spv` dosyasından) — planın en riskli varsayımı baştan doğrulanmış oldu.
  - Tekrar ETMEYECEĞİMİZ hatalar: (a) `graph.rs::add_operation` her çağrıda
    pipeline+layout+descriptor-set'i sıfırdan kuruyor, hiç cache yok — gerçek
    model ölçeğinde (step başına yüzlerce node) bu tek başına performansı
    öldürür, son commit mesajının ("vulkano bana engel oluyor") sebeplerinden
    biri muhtemelen bu; (b) bariyer YOK — `ExecutableGraph::execute()` her
    graph'tan sonra fence wait + `unsafe { device.wait_idle() }` ile TAM
    senkron oluyor, yani filuplex kendi bariyer sorununu hiç çözmedi, aynı
    FORCE_SYNC kabalığını graph seviyesinde tekrarladı; (c) shader'lar
    `vulkano_shaders::shader!` makrosuyla elle GLSL yazılmış (`shaders.rs`) —
    ikinci bir shader dili bakım yükü, tam da naga ile ortadan kaldırmak
    istediğimiz şey.
  - Sonuç: filuplex'ten kodu değil, "SPIR-V runtime yükleme çalışıyor"
    kanıtını ve "cache'siz + bariyersiz asla hızlı/doğru olmaz" dersini
    alıyoruz.
- Durum: sadece boş `backends/vulkano.rs` + `vulkano` feature flag (Cargo.toml,
  `vulkano` crate henüz dependency olarak eklenmedi) atıldı. İlk gerçek adım:
  wilupgu'ya hiç dokunmadan, bağımsız bir scratch'te "WGSL string → naga →
  SPIR-V → vulkano tek dispatch" zincirinin uçtan uca çalıştığını kanıtlamak.

**2) Gerçek paralel CPU backend (`backends/rayon.rs`).**
- Mevcut `CpuBackend` bilinçli olarak tek-thread (test determinism için) —
  kalıyor, dokunulmadı.
- Yeni `RayonBackend`: `CpuBackend` ile birebir aynı buffer/pool mantığı
  (`CpuBuffer = Arc<Mutex<Vec<u8>>>`), tek fark `build_node`'un `shader.cpu`
  yerine yeni `shader.rayon: Option<fn(&[CpuBinding])>` alanına bakması.
  Paralellik node-dispatch seviyesinde değil (node'lar genelde ardışık
  bağımlı) — her kernel'in KENDİ gövdesi içinde (`par_chunks_mut` vb.) olacak,
  bu yüzden imza `cpu` ile birebir aynı kalabildi.
- Durum: iskelet atıldı — `Shader` struct'ına `rayon` alanı + wilupgu'nun 12 +
  sequexa-core'un 30 static'ine `rayon: None,` eklendi, `RayonBackend` tam
  `Backend` impl'i ama her kernel şu an `panic!("no rayon impl yet")` veriyor.
  Gerçek paralel kernel gövdeleri (matmul, adamw, vb.) tek tek, ayrı işler
  olarak eklenecek — 200+ çekirdekli sunucularda gerçek kazanç burada olacak.

**3) Backend trait genişletmesi + CUDA'nın ikiye bölünmesi (henüz
tasarlanmadı, sadece niyet).**
- `cuda.rs` (1189 satır) `wgpu.rs`'ten (370 satır) çok şişkin çünkü ikisi aynı
  işi yapmıyor: cuBLAS GEMM/GEMM_EX entegrasyonu + CUDA graph capture/replay +
  dtype-generic `Gemm<T>` hepsi `cuda.rs`'te, `wgpu.rs`'te hiçbiri yok (matmul
  elle WGSL). Plan: `cuda.rs`'i `wgpu.rs` kadar ince, BLAS'sız, generic
  dispatch'e indirmek; BLAS'lı yol ayrı bir `cuda-blas.rs` backend'i olsun —
  iki backend, aynı CUDA runtime, farklı matmul stratejisi.
- Backend trait'inin kendisinin de "daha dinamikleştirilebilir" olduğu
  düşünülüyor — somut madde yok, `backend.rs`'i birlikte açıp geçtiğimizde
  netleşecek.

## 🔵 Feat'ler

**F4** — ember CUDA shader'ları.
**F6** — Quantization (NNUE int8 ölçekleme) — ember entegrasyonuyla birlikte
yapılacak: NNUE zaten quantization-aware eğitim istiyor, zemin oraya kurulur.

## 🧭 Strateji notları (yeni bug çıkmasın diye)

1. **Kontratları yoruma değil, teste bağla.** Output/Accumulate etiket
hataları yakalanmadı çünkü etiket sadece beyan. Output etiketli her buffer'ı
dispatch öncesi NaN/çöple doldurup çıktıyı kontrol eden tek bir "canary"
testi bu sınıfı otomatik yakalar. Bir kere yaz, her yeni kernel bedavaya
taransın.
2. **Kopya yüzeyini küçült.** Bug'ların önemli kısmı ikiz implementasyonların
ayrışması (WGSL'de eksik barrier, CUDA'da var; guard bir kernelde var
birinde yok). Eksik CPU impl'leri doldurulursa "her kernel × her backend ×
CPU referansı" mekanik parity matrisi kurulur — o matris varken ikizler
sessizce ayrışamaz.
3. **Formül yazarken köşeleri aynı anda test et.** Schedule clamp, boş
prompt, dataset underflow — hepsi aynı sınıf: parametrenin sınır değeri.
Parametre alan her fonksiyonun testine t=0, t=sınır, t=sınır+1 satırlarını
eklemek bu sınıfı neredeyse bitirir. Maliyeti dakikalar.
4. **Periyodik taramayı ritüelleştir.** Her N commit'te ya da her büyük
refactor sonrası bağımsız okuma turu — solo geliştiricinin code review'u
budur.
5. **Panik satırını doğrulamadan teori kurma (B13'ten ders).** Bir hata
mesajındaki dosya:satır'ı hangi fonksiyona ait olduğunu okumadan "muhtemelen
X" demek, yanlış fonksiyonlara yama yapmaya götürür. Önce satırın gerçekten
hangi çağrıya ait olduğunu doğrula, sonra teori kur.

Sonrası: continued pretraining (yaklaşan ~10 günlük koşu; Big Refactor o
sırada tasarlanır) → chat fine-tuning.
