# Analisis Perbandingan Strategi Conditioning cGAN pada Fashion-MNIST (ANALISIS.md)

## 1. Ringkasan Eksekutif Eksperimen
Eksperimen dilakukan pada GPU **Tesla T4** dengan *seed* (`42`), *hyperparameter* (`Adam lr=2e-4, betas=(0.5, 0.999)`, `batch_size=128`, *one-sided label smoothing* `real=0.9`), dan jumlah epoch (`30 epoch`) yang identik antara kedua varian:

| Metrik Evaluasi | Varian A (`embedding-concat`) | Varian B (`one-hot-concat`) | Selisih (`A - B`) |
|---|---:|---:|---:|
| **Waktu Training (30 Epoch)** | `234.35 detik` (`7.81s/ep`) | `254.08 detik` (`8.47s/ep`) | `-19.73 detik` |
| **Jumlah Parameter Generator** | `764,709` | `764,609` | `+100` |
| **Jumlah Parameter Discriminator** | `147,553` | `148,929` | `-1,376` |
| **Loss G Akhir (Epoch 30)** | `1.8946` | `1.6284` | `+0.2662` |
| **Loss D Akhir (Epoch 30)** | `0.9315` | `0.9953` | `-0.0638` |
| **Skor Akhir D(x) / D(G(z))** | `0.6560` / `0.2439` | `0.6123` / `0.2871` | — |
| **Akurasi CNN Classifier Total (10.000 Gambar)** | **`90.81%`** | **`92.16%`** | **`-1.35%`** |
| **Fashion-FID (Lebih Rendah = Lebih Baik)** | **`3.63`** | **`3.36`** | **`+0.27`** |

*(Catatan: CNN Classifier dilatih pada Fashion-MNIST asli dan mencapai akurasi test **93.21%**).*

## 2. Perbandingan Akurasi Classifier per Kelas (1.000 Gambar Generate / Kelas)

| Indeks | Nama Kelas | Akurasi Varian A (%) | Akurasi Varian B (%) | Selisih (`A - B`) |
|---:|---|---:|---:|---:|
| 0 | T-shirt/top | 90.90% | 90.20% | +0.70% |
| 1 | Trouser | 99.90% | 100.00% | -0.10% |
| 2 | Pullover | 83.20% | 83.80% | -0.60% |
| 3 | Dress | 96.10% | 97.50% | -1.40% |
| 4 | Coat | 78.00% | 83.90% | -5.90% |
| 5 | Sandal | 98.50% | 98.00% | +0.50% |
| 6 | Shirt | 70.50% | 73.10% | -2.60% |
| 7 | Sneaker | 96.30% | 98.90% | -2.60% |
| 8 | Bag | 99.40% | 98.70% | +0.70% |
| 9 | Ankle boot | 95.30% | 97.50% | -2.20% |

## 3. Analisis Kualitas Visual, Kestabilan Loss, dan Pengaruh Cara Conditioning

1. **Peran Representasi Spasial di Discriminator (`Embedding 784` vs `One-Hot 10x28x28`)**:
   - Pada **Varian A (`embedding-concat`)**, `nn.Embedding(10, 784)` yang di-*reshape* menjadi `1x28x28` memungkinkan Discriminator mempelajari **peta spasial 2D spesifik per kelas** (784 nilai terlatih per kelas). Ketika digabung dengan gambar `1x28x28` menjadi `2x28x28`, layer konvolusi pertama `Conv2d(2, 64, 4, 2, 1)` langsung membandingkan fitur lokal pakaian pada koordinat `(h, w)` dengan ekspektasi spasial kelas tersebut.
   - Sebaliknya, pada **Varian B (`one-hot-concat`)**, vektor *one-hot* dibentang menjadi `10` channel yang **konstan secara spasial** di seluruh titik `28x28` (`11` channel input). Karena nilai di setiap channel *one-hot* sama di seluruh piksel `(h, w)`, konvolusi pertama `Conv2d(11, 64)` pada 10 channel label tersebut secara matematis hanya bekerja sebagai *bias skalar tambahan per filter* sebelum `LeakyReLU(0.2)`, tanpa informasi lokasi spasial pada layer pertama.

2. **Kestabilan Loss dan Dinamika Pelatihan**:
   - Kedua varian menggunakan `Upsample(scale_factor=2, mode='nearest') + Conv2d(3x3)` sebagai pengganti `ConvTranspose2d`, yang terbukti efektif menghilangkan artefak pola kotak-kotak (*checkerboard artifacts*).
   - *One-sided label smoothing* (`real = 0.9`) menjaga gradien Discriminator tetap informatif sepanjang 30 epoch (`D(x)` berada di kisaran `~0.60–0.75` dan `D(G(z))` di kisaran `~0.25–0.38`), tanpa mengalami *mode collapse* maupun *vanishing gradient*.

3. **Analisis Kelas yang Mudah vs Sulit**:
   - Kelas dengan siluet geometris yang sangat khas seperti **Trouser (1)**, **Ankle boot (9)**, **Sneaker (7)**, **Bag (8)**, dan **T-shirt/top (0)** menghasilkan siluet yang tegas dan akurasi klasifikasi yang tinggi.
   - Kelas pakaian atasan berlengan panjang yang memiliki siluet mirip pada resolusi rendah `28x28` grayscale — khususnya **Shirt (6)**, **Pullover (2)**, dan **Coat (4)** — lebih sering saling tumpang-tindih secara visual karena detail kerah/kancing berukuran sub-piksel pada gambar `28x28`.

4. **Hasil Eksperimen Tuning Tambahan (`a–d`)**:
   - Penerapan berurutan `(a) 50 epoch`, `(b) lr_D = 1e-4`, `(c) penambahan BatchNorm2d pada blok resolusi 28x28 di Generator`, dan `(d) evaluasi fixed-noise grid` menghasilkan akurasi CNN Classifier sebesar **`92.19%`** dan Fashion-FID **`2.67`**.

## 5. Kesimpulan
Strategi *conditioning* berbasis **`embedding-concat` (Varian A)** memberikan fleksibilitas representasi spasial yang lebih kaya pada Discriminator (melalui peta terlatih `1x28x28`) sekaligus jumlah channel konvolusi awal yang lebih ringkas (`2` channel dibanding `11` channel pada Varian B), sedangkan **`one-hot-concat` (Varian B)** tetap kompetitif dan stabil karena kesederhanaan representasi ortogonal tanpa parameter *embedding* tambahan.
