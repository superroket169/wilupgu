# Bekleyen işler

Kapsam: akasha-core + wilupgu + ember. Yalnız YAPILMAMIŞ maddeler; biten iş
buradan silinir (tarihçe git log'da). Sıra: doğruluk → hız → tasarım → feat.

## 🟠 Hız

**B13** — Cuda graph capture'u çok dengesiz çalıştığı için artık opsiyonel çalışıyor.
cuda graph dan resmen pes edilmiştir. kod hala bulunacaktır ancak kullanılması zorlanmayacaktır.
Performans konusunda karşılaştırmalar yapılacaktır. şimdi tam net olmamak ile birlikte sadece cpu tarafında
bir fazla kullanım söz konusu olabilir. ileride netleştirilecektir.

**B9** — CUDA decode: her matmul dispatch'inde bloklayan dtoh (cuda.rs
gemm_meta_u32): capture dışında her cuBLAS çağrısı meta'yı device'tan senkron
çeker. Decode graph capture edilmiyor → token başına ~61 matmul × bloklayan
kopya. Decode'un matmul metaları sabit; matmul-family capture dışında da
cached_meta okusa maliyet kalkar. Dikkat: gerçekten dinamik meta'lı bir cuBLAS
çağrısı varsa ona opt-out gerekir. **Not:** akasha ARCHITECTURE fikir
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

**1) Flash attention register spill — DÜZELTİLDİ (lokal, commit'lenmedi).**
`flash_attention.wgsl` / `flash_attention_bwd_dq.wgsl` /
`flash_attention_bwd_dkdv.wgsl` üçünde de `array<f32, MAX_HEAD_DIM>` tipinde
per-thread accumulator, döngü sınırı runtime Meta değeri (`m.head_dim`) olduğu
için derleyici tarafından unroll edilemiyor, register yerine VRAM-backed
scratch belleğe spill oluyor (RADV hang dump'ında doğrulandı: VGPRs=16,
Scratch=64KB/wave). Bu hem ciddi yavaşlık hem de (dkdv'de iki dizi olduğu
için) gerçek bir `radv/amdgpu: GPU hang` (`ring gfx_0.0.0 timeout`) sebebiydi.
Fix: üç dosyada da `m.head_dim` yerine bu modelin sabit `HEAD_DIM: u32 = 64u`
compile-time sabiti kullanılıyor (config.rs'teki `DIM/NUM_HEADS` ile aynı).
**Eksik:** bu, `emit.rs::flash_attention_validation`'ın farklı head_dim'lerle
test ettiği boundary-value testlerini kıracak (pipeline cache tek static WGSL
kaynağına bağlı, per-head_dim varyant yok) — production'a girmeden önce ya
assert + tek-head_dim kabulü ya da head_dim'e göre shader üretimi gerekiyor.
B11'in (a) maddesiyle aynı kök alanı (tiling yokluğu) ama farklı somut sorun.

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

**Sıradaki adımlar** (öncelik sırasıyla, hiçbiri bu oturumda tamamlanmadı):
- Kısa vade: `WILUPGU_FORCE_SYNC=1` ile gerçek bir eğitim koşusu başlat
  (doğru ama yavaş — her node ayrı submit+wait, pipelining tamamen kayboluyor).
- Orta vade: tüm node'lar yerine sadece gerekli 1-2 sınırda senkron
  (bisection ile minimal bariyer noktasını bul, çoğu hızı geri kazan).
- Uzun vade: `wgpu` 0.19.4 → 30.0.0 upgrade — muhtemel kalıcı çözüm, ayrı
  bir oturumluk iş.
- Flash attention head_dim sabitlemesinin test-genelliği sorunu (madde 1)
  ayrıca çözülmeli, production'a girmeden önce.

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
