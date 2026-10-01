# TODO

Kapsam: sequexa-core + wilupgu ortak bekleyen iş listesi. Yalnız
YAPILMAMIŞ maddeler — biten iş buradan silinir, tarihçe git log'da.

## Hız

- **Decode'u cuBLAS'sızlaştırmak** — `gemv.wgsl`/`gemv_add.wgsl`'in CUDA C
  çevirisi yazılıp GEMV/GEMV_ADD `CudaShape::Custom`(cuBLAS)'tan
  `Generic`'e geçer. Decode'daki tüm matmul'lar m=1 → GEMV olduğundan
  decode graph'ında hiç cuBLAS kalmaz: dispatch-başı bloklayan meta dtoh'u
  (CUDA decode'un her matmul çağrısında yaptığı senkron kopya) kökten
  kalkar VE decode graph'ı capture edilebilir hale gelir. İlk adım: güncel
  kodda CUDA↔Vulkan'ı yeniden ölçmek — eldeki 80-vs-31 step/dk sayısı
  pooling/flash öncesi dönemden, güncel değil.
- **Flash attention: shared memory/tiling yok** — thread-per-(row,head)
  tasarımı K/V'yi her satır için global'den tekrar okuyor.
- **wgpu otomatik bariyer bug'ı** — çok-node'lu tek compute pass'te
  dispatch'ler arası bariyer eksik/gecikmeli, `Parent device is lost` ile
  çöküyor. Tek çalışan workaround `WILUPGU_FORCE_SYNC=1` (doğru ama yavaş —
  her node ayrı submit+wait). wgpu 0.19→30 upgrade denendi, çözmedi. Uzun
  vadeli çözüm aşağıdaki native Vulkan backend maddesi.

## Tasarım (Büyük Wilupgu Refactoru)

- **Native Vulkan backend** (`backends/vulkano.rs`, şu an boş dosya) —
  yukarıdaki wgpu bariyer bug'ını çözmek için. Zincir:
  `naga::front::wgsl::parse_str` → `naga::valid::Validator` →
  `naga::back::spv::write_vec` (WGSL→SPIR-V, wgpu'nun içeride zaten yaptığı
  şey) → `vulkano::ShaderModule::new`. Bariyer: node'ların
  `Binding`/`TensorMode`'undan gerçek RAW/WAW hazard'larını çıkarıp elle
  `vkCmdPipelineBarrier` basmak. İlk gerçek adım: wilupgu'ya dokunmadan
  bağımsız bir scratch'te "WGSL → naga → SPIR-V → vulkano tek dispatch"
  zincirini uçtan uca kanıtlamak.
- **Gerçek paralel CPU backend** (`backends/rayon.rs`) — iskelet hazır
  (`Shader.rayon` alanı + tam `Backend` impl'i), ama her kernel şu an
  `panic!` veriyor. Gerçek paralel kernel gövdeleri (matmul, adamw, vb.)
  tek tek eklenecek — 200+ çekirdekli sunucularda gerçek kazanç burada.
  Mevcut `CpuBackend` bilinçli tek-thread kalıyor (test determinism).
- **Backend trait genişletmesi + CUDA'nın ikiye bölünmesi** — `cuda.rs`
  (1189 satır) `wgpu.rs`'ten (370 satır) şişkin çünkü cuBLAS GEMM/GEMM_EX +
  CUDA graph capture/replay + dtype-generic `Gemm<T>` hepsi orada. Plan:
  `cuda.rs`'i `wgpu.rs` kadar ince, BLAS'sız, generic dispatch'e indirmek;
  BLAS'lı yol ayrı bir `cuda-blas.rs` backend'i olsun. `Backend` trait'inin
  kendisinin de daha dinamikleştirilebilir olduğu düşünülüyor — somut
  madde yok, `backend.rs`'i birlikte açıp geçtiğimizde netleşecek.
- **Flash attention head_dim'i WGSL `override` ile genelleştirmek** —
  bugün `const HEAD_DIM: u32 = 64u` hardcode'lu (register-spill/RADV-hang
  fix). Gerçek genel çözüm WGSL'nin pipeline-overridable constant'ı, ama
  wilupgu'nun `Shader`/pipeline-cache API'sinde bu kavram hiç yok.

## Küçük

- `backends/wgpu.rs` — f16 ve diğer dtype'lar için destek yok.
- `shaders/wgsl/fwd/rope.wgsl` — `inv_freq` precompute edilerek
  optimize edilebilir.
- GPU'nun eğitimde CPU'dan gerçekten bağımsız olduğunun doğrulanması —
  steady-state train loop'ta kasıtlı trafik dışında gizli host
  senkronu/kopyası olmadığını kanıtla.

## Uzak gelecek (parked)

- **ROCm/HIP backend** — gerçek AMD backend (rocBLAS tabanlı,
  `CudaBackend`'e paralel). wgpu/Vulkan zaten RADV üzerinden AMD'yi
  karşıladığı için düşük öncelikli. Canlı crate adayları: `cubecl-hip-sys`,
  `rocm-rs`.
- **`dispatch_generic` tarzı factoring** — `impl Backend for CudaBackend`'in
  ~280 satırlık trait-impl çekirdeğini küçültmek, ROCm ikinci tüketici
  olmadan önce.
- **WASM/WebGPU tarayıcı demosu** — wgpu backend'in doğal bir yan ürünü.
- **Geniş quantization (int8/int4 inference)** — decode'un bant genişliği
  sınırlı olması yüzünden genel bir iGPU kaldıracı; yukarıdaki NNUE-bağlı
  quantization'dan ayrı, daha geniş kapsamlı.

## Strateji notları (yeni bug çıkmasın diye)

1. **Kontratları yoruma değil, teste bağla.** Output/Accumulate etiket
   hataları yakalanmadı çünkü etiket sadece beyan. Output etiketli her
   buffer'ı dispatch öncesi NaN/çöple doldurup çıktıyı kontrol eden tek bir
   "canary" testi bu sınıfı otomatik yakalar. Bir kere yaz, her yeni kernel
   bedavaya taransın.
2. **Kopya yüzeyini küçült.** Bug'ların önemli kısmı ikiz
   implementasyonların ayrışması (WGSL'de eksik barrier, CUDA'da var; guard
   bir kernelde var birinde yok). Eksik CPU impl'leri doldurulursa "her
   kernel × her backend × CPU referansı" mekanik parity matrisi kurulur —
   o matris varken ikizler sessizce ayrışamaz.
3. **Formül yazarken köşeleri aynı anda test et.** Schedule clamp, boş
   prompt, dataset underflow — hepsi aynı sınıf: parametrenin sınır değeri.
   Parametre alan her fonksiyonun testine t=0, t=sınır, t=sınır+1
   satırlarını eklemek bu sınıfı neredeyse bitirir. Maliyeti dakikalar.
4. **Periyodik taramayı ritüelleştir.** Her N commit'te ya da her büyük
   refactor sonrası bağımsız okuma turu — solo geliştiricinin code review'u
   budur.
5. **Panik satırını doğrulamadan teori kurma.** Bir hata mesajındaki
   dosya:satır'ı hangi fonksiyona ait olduğunu okumadan "muhtemelen X"
   demek, yanlış fonksiyonlara yama yapmaya götürür. Önce satırın gerçekten
   hangi çağrıya ait olduğunu doğrula, sonra teori kur.
