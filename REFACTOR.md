# Bugfix

Kapsam: akasha-core + wilupgu, ember sonda. Kademeler küçük — her biri tek
oturumda bitecek boyutta. Sıra: doğruluk → crash → hız → VRAM → temizlik → feat.

## 🟢 Eğitim kalitesi ayarları

**E1** ✅ (2026-07-16) — Weight decay norm ağırlıkları ve embedding'den
çıkarıldı: collect_trainable_params artık (weight, grad, decay) üçlüsü
üretiyor (bayrak parametrelerin adlandırıldığı tek yerde); AdamW::new iki
ConstCfg tutuyor (decay'li/decay'siz) ve node başına uygununu bağlıyor.
Sıra sözleşmesi ve moments düzeni değişmedi; dış trainable_params() API'si
çift dönmeye devam ediyor (diagnose.rs etkilenmedi).
**E2** ✅ (2026-07-16) — out_proj / ffn_down init std'sine 1/√(2L) çarpanı
(weights.rs::random; GPT-2 pratiği — residual akışına yazan 2L projeksiyonun
birikimi derinlikten bağımsız kalsın). Yalnız sıfırdan init'i etkiler;
mevcut checkpoint'ten devam eden koşulara etkisi yok.

## 🔵 Yeni feat'ler (en sona)

**F1** — Gerçek batch size'ı decode/prefill/causal_attention'a tamamla.
**F2** ✅ (2026-07-16) — Streaming dataset: data.rs yeniden yazıldı. Raw
corpus bir kez chunk-chunk (8MB, UTF-8 + satır/kelime sınırında kesim)
tokenize edilip 16M-token'lık raw u32 LE shard dosyalarına yazılıyor
(data/train_shards/; dizin varsa yeniden tokenize edilmez). Eğitimde en
fazla 4 shard bellekte (~256MB), her 256 batch'te biri rastgele soğuk
shard'la değişiyor; pencere-sayısı-ağırlıklı örnekleme. Chunk mekaniği
encode closure'ı aldığı için tokenizersız test edilebilir — 3 host testi:
losslessness (çok byte'lı char'lar sınırda), batch tutarlılığı + rotasyon,
B8 paniği. 50M-char truncation tarihe karıştı.
**F3** — Docs pass.
**F4** — ember CUDA shader'ları (K6'dan sonra).
**F5** — ember: ClippedReLU tek copy-clamp kernel'i; mse_loss graph'ını
her train_step'te kurmak yerine yeniden kullan.
**F6** — Quantization zemini (NNUE int8 ölçekleme).

### 🔴 Doğruluk

**B1** — K6 doğrulandı: ember AdamW hâlâ kırık. ember/src/optim/adamw.rs:83-89
ADAMW'ye 6 slot bağlıyor; layout 7 slot ([InOut, Input, InOut, InOut, Meta,
Input, Meta], wilupgu builtin/mod.rs:157). Slot 5'e StepConfig'i Meta olarak
veriyor ama layout Input (ScheduleState{step,lr}) bekliyor → add_node mode
assert'inde panik; slot 6 (ConstCfg) hiç yok. Trainer::new kurulurken çöker,
smoke test geçemez.
**B2** ✅ (2026-07-15) — rmsnorm_bwd.wgsl'de eksik barrier (workgroup
yarışı): `reduce()` sonundaki barrier yazımları bitiriyordu ama partial[0]
okumalarını değil; hızlı thread bir sonraki `partial[tid] = ...` ile diziyi
erken ezebiliyordu → nadiren yanlış dX. Fix: sonuç yerel değişkene alınıp
okuma sonrası workgroupBarrier() (CUDA ikizi zaten doğruydu).
**B3** — T1 taramasının sonucu, üç kernel Output etiketli ama accumulate:
EMBEDDING_BWD (akasha shaders/mod.rs:26, atomik += — WGSL CAS loop / CUDA
atomicAdd), RMSNORM_WEIGHT_BWD (mod.rs:287, dWeight[i] += acc), ember
BIAS_ADD_BWD (dBias[j] += acc). Bugün davranış doğru (trainer zero'luyor)
ama Output kontratı "pool çöpü asla sızmaz" der — taze pool buffer'ı
bağlanırsa çöp grad üretir, T1'in dokümantasyon değeri kaybolur. (Kontrol
edildi: flash bwd dq/dkdv gerçekten tam overwrite, Output etiketi doğru.)
**B4** ✅ (2026-07-15) — Cosine schedule progress dört kopyada da
(adamw_schedule.wgsl, cuda_kernels.rs, cpu_kernels.rs, config.rs host
formülü) aynı desenle clamp'lendi: max_steps > warmup_steps ise
progress.min(1.0), değilse (0/0 durumu) progress = 1.0 → lr_min.
Host formülüne sınır testi eklendi (config.rs::tests::cosine_lr_boundaries:
t=0, warmup, max, max+N, max==warmup). CUDA kopyası bu makinede
derlenemiyor — nvidia makinede bir tur bekliyor.
**B5** ✅ (2026-07-16) — V3 checkpoint: AKV3 = weights + AdamW m/v momentleri
+ schedule_step + train_step (dosya içinde; resume artık dosya adına muhtaç
değil). V1/V2 kütüphaneden tamamen söküldü — legacy okuyucular yalnız yeni
bin/migrate_checkpoint_v3.rs'te (v1 VE v2 okur, bitwise verify'lı; bu
makinedeki model_final.bin.v2.bin → model_final.v3.bin migre edildi, 75
tensör doğrulandı). main.rs: resume model_step_* → model_final.v3.bin →
sıfırdan sırasıyla; final kayıt model_final.v3.bin'e (model_final.bin v1
anısı, asla yazılmaz). Bekçi test: v3_save_load_roundtrip. Kalan ideal:
RNG/data cursor — F2 streaming dataset kendi cursor'ını getirince oraya.
Not: eski model_step_*.bin (v2) dosyaları artık yüklenmez — migre et ya da
sil.
**B6** ✅ (2026-07-15) — Tarif düzeltmesi: "seq_len'lik pencere rows
buffer'ına" mekanizması güncel kodda yok — train_step'in penceresi
cross_entropy.seq_len = rows (batch*seq) boyutlu ve giriş uzunlukları
zaten assert'li (tarama yerel `seq_len` adına aldanmış). Kapatılan gerçek
boşluklar: (1) arg batch_size ile cfg.batch_size birlikte >1 (dokümante
tanımsız kombinasyon) artık train_step'te assert'le reddediliyor;
(2) Trainer::new, caller'ın verdiği input_tokens tensörünün rows token
tuttuğunu assert ediyor — bayat-satır riskinin asıl kaynağı buydu.
**B7** — generate() boş prompt + dolu cache → panik (inference.rs:267-275):
prefill yolu EmptyPrompt dönerken, resumed-cache yolunda prompt_tokens boşsa
`last = Vec::new()` kalır ve sample_token(&[], ...) idx[0]'da paniker.
**B8** ✅ (2026-07-16) — F2 ile birlikte kapandı: seq_len+1'den küçük
shard'lar uyarıyla elenir, hiç kullanılabilir shard kalmazsa anlamlı
assert mesajı (test: tiny_corpus_panics_instead_of_underflowing).

### 🟠 Hız

**B9** — CUDA decode: her matmul dispatch'inde bloklayan dtoh
(cuda.rs:226-236 gemm_meta_u32): capture dışında her cuBLAS çağrısı meta'yı
device'tan senkron çeker. Decode graph capture edilmiyor → token başına ~61
matmul × bloklayan kopya + stream stall; CUDA decode gecikmesini bu domine
eder. Oysa decode'un matmul metaları sabit (m,n,k değişmiyor; değişenler
attention/rope/cache metaları, onlar cuBLAS değil). build_node zaten
cached_meta dolduruyor; matmul-family capture dışında da cached_meta okusa
maliyet kalkar. Dikkat: gerçekten dinamik meta'lı bir cuBLAS çağrısı varsa
ona opt-out gerekir.
**B10** — Prefill tüm prompt satırları için logits hesaplıyor
(inference.rs:136-156): [prompt_len, 50257] matmul + buffer (512 token'da
~100MB VRAM) + tamamının host'a kopyası — sadece son satır kullanılıyor.
final_out'un son satırına m=1 GEMV → lm_head maliyeti prompt_len kat düşer,
100MB buffer ve dev host kopyası kalkar. Muhtemelen en yüksek getirili
tekil optimizasyon.
**B11** — Flash attention verimlilik notları (doğruluk tamam): (a)
thread-per-(row,head) tasarımı shared memory/tiling kullanmıyor; K/V her
satır için global'den tekrar okunuyor. (b) bwd_dkdv içteki döngüde d_i'yi
her (col,i) çifti için yeniden hesaplıyor — FlashAttention-2 gibi tek
geçişte D[i] = Σ dO·O precompute edilirse bwd'den koca bir head_dim
döngüsü çıkar.
**B12** — Prefill her çağrıda graph'ı ve tüm ara buffer'ları yeniden kuruyor
(inference.rs:83-152). Pool yumuşatıyor ama prompt başına build + upload
maliyeti var; uzunlukları bucket'layıp graph cache'lemek mümkün.
**B13** — checkpoint::save tüm parametreleri tek seferde host'a topluyor;
V3 momentleri de eklediği için tepe ~3× büyüdü (162M model için ~2GB
Vec<Vec<f32>> + bincode). Tensor-tensor streaming yazım RAM spike'ını da
duraklamayı da azaltır — V3 sonrası önceliği arttı.

### 🟡 Tasarım / tutarlılık

**B14** — RMSNorm eps ikiliği: layers.rs:144 eps=1e-5 hardcode; inference
yolu cfg.norm_eps kullanıyor (inference_graphs.rs:58). Bugün ikisi de 1e-5
ama biri değişirse train/inference sessizce ayrışır.
**B15** — T3 asimetrisi: %4 assert sadece CUDA alloc'ta (cuda.rs:598);
wgpu/cpu alloc'ta yok. wgpu'da tek yakalanma yeri wgpu'nun kendi validation
hatası olur.
**B16** — grid256 1D limiti (emit.rs:679): silu/add/residual_add eleman
sayısı 16.7M'i (65535×256) aşarsa sessizce dispatch dışı kalır. Bugün
güvenli (max ~3.1M) ama model büyüyünce görünmez mayın; grid256_2d zaten
var, tutarlı kullanmak ucuz.
**B17** — embedding.wgsl geçersiz token id'de satırı hiç yazmıyor ama Output
("tamamen üzerine yazılır") etiketli — pool'lu out buffer'da çöp hidden
state sızabilir. Bugün teorik (tokenlar hep vocab içi); ya satırı sıfırla
ya yorumla belgele.
**B18** — ember bias_add.wgsl'de bounds guard yok — T4 eklerken atlanmış.
Pow2 pool sayesinde bugün zararsız ama projenin kendi konvansiyonuna aykırı.
**B19** — CpuBuffer = Arc<Mutex<Vec<u8>>> + cast_slice (cpu.rs:62,
cpu_kernels.rs:16): Vec<u8>'in f32 hizası garanti değil; cast_slice
hizasızlıkta panikler (pratikte allocator 16 byte veriyor diye çalışıyor).
bytemuck::pod_collect_to_vec ya da Vec<f32> tutmak latent paniği kapatır.
**B20** — Ufaklıklar: generate'te EOS=50256 hardcode (config'e taşınabilir);
zero_grads_graph transient'leri ikinci kez zero'luyor (zararsız israf);
ember shaders/mod.rs başındaki "mistakenly file" yorumu kalmış;
wilupgu::CAUSAL_MASK H5 sonrası üretim yolunda kullanıcısız (sadece parity
testi kullanıyor).

## 🧭 Strateji notları (yeni bug çıkmasın diye)

1. **Kontratları yoruma değil, teste bağla.** Output/Accumulate etiket
hataları yakalanmadı çünkü etiket sadece beyan. Output etiketli her buffer'ı
dispatch öncesi NaN/çöple doldurup çıktıyı kontrol eden tek bir "canary"
testi EMBEDDING_BWD ve RMSNORM_WEIGHT_BWD'yi otomatik yakalardı. Bir kere
yaz, her yeni kernel bedavaya taransın.
2. **Kopya yüzeyini küçült.** Bug'ların önemli kısmı ikiz implementasyonların
ayrışması (WGSL'de eksik barrier, CUDA'da var; T4 guard'ı bir kernelde var
birinde yok). Eksik CPU impl'leri (SILU_BWD, RMSNORM_BWD vb.) doldurulursa
"her kernel × her backend × CPU referansı" mekanik parity matrisi kurulur —
backend_parity.rs'in genelleştirilmişi. O matris varken ikizler sessizce
ayrışamaz.
3. **Formül yazarken köşeleri aynı anda test et.** Schedule clamp, boş
prompt, dataset underflow — hepsi aynı sınıf: parametrenin sınır değeri.
(warmup, max_steps, len) gibi parametre alan her fonksiyonun testine t=0,
t=sınır, t=sınır+1 satırlarını eklemek bu sınıfı neredeyse bitirir.
Maliyeti dakikalar.
4. **Periyodik taramayı ritüelleştir.** T1'in "akasha shader'ları taransın"
notu bugünkü bulguları önceden haber vermişti. Her N commit'te ya da her
büyük refactor sonrası bağımsız okuma turu — solo geliştiricinin code
review'u budur.

Sonrası: continued pretraining → chat fine-tuning (E1/E2 + B4/B5/B6 bitmeden
continued pretraining'e başlama).
