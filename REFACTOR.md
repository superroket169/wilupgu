# Bekleyen işler

Kapsam: akasha-core + wilupgu + ember. Yalnız YAPILMAMIŞ maddeler; biten iş
buradan silinir (tarihçe git log'da). Sıra: doğruluk → hız → tasarım → feat.

## 🟠 Hız

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

Sonrası: continued pretraining (yaklaşan ~10 günlük koşu; Big Refactor o
sırada tasarlanır) → chat fine-tuning.
