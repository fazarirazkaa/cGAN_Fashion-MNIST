# Log Interaksi & Dokumentasi LLM 

---

## Sesi 1 — Modul 1: Inisiasi Lingkungan GPU Kaggle, Data Pipeline Fashion-MNIST, & Fallback Offline

```text
LOG INTERAKSI LLM - KELOMPOK [FashionGAN]
• Tanggal Sesi: 20/10/2026
• Target Modul: Setup Lingkungan GPU Kaggle, DataLoader Fashion-MNIST Normalisasi [-1, 1], & Mekanisme Fallback CSV Offline
• Tautan Sesi Chat: 
- Claude : https://claude.ai/share/448920af-5a33-420c-86f8-95c72ac086b5
- Gemini Argon : file markdown LLM_LOG.md dan LOG_INTERAKSI_LLM.md 
```

### KRONOLOGI PROMPT & RESPONS:

#### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok**:
  > *"Kamu adalah AI engineer yang membangun dan menjalankan proyek Conditional GAN (cGAN) klasik pada Fashion-MNIST di Kaggle Notebook dengan GPU. Output disimpan ke `/kaggle/working` dengan membuat folder `models/`, `outputs/`, dan `app/`. Cek `torch.cuda.is_available()`, nama GPU, dan memori VRAM. Jika GPU tidak aktif, hentikan eksekusi dengan `RuntimeError` dan beri tahu cara mengaktifkannya (`Settings > Accelerator`); jangan lanjut di CPU. Gunakan 1 GPU (`cuda:0`) tanpa `DataParallel`, `DataLoader` dengan `num_workers=2, pin_memory=True`, `batch_size=128`, dan `seed=42`. Muat `torchvision.datasets.FashionMNIST(download=True)`, normalisasi ke `[-1, 1]`, bentuk tensor `1x28x28`. Jika unduhan gagal karena kendala internet, cari file CSV di `/kaggle/input` menggunakan `os.walk` dan reshape 784 piksel ke `1x28x28`. Simpan 10 nama kelas resmi ke `models/classes.json` dan simpan grid contoh gambar asli per kelas (10 baris x 8 kolom) ke `outputs/sample_real.png`."*

- **Alasan Khusus**:
  1. Memastikan pelatihan cGAN wajib berjalan di akselerator GPU (`Tesla T4` `cuda:0`) dan langsung berhenti (*fail-fast*) apabila sesi Kaggle tidak sengaja berjalan di CPU.
  2. Menyamakan rentang piksel input `[-1, 1]` dengan rentang fungsi aktivasi `Tanh` pada layer terakhir Generator.
  3. Menjamin notebook tetap *reproducible* dan tahan gagal (*fault-tolerant*) meskipun koneksi unduhan `torchvision` terputus di tengah sesi Kaggle dengan menyediakan *fallback* pembacaan CSV dari `/kaggle/input`.

- **Output Model**:
  LLM menghasilkan skrip inisiasi lengkap beserta kelas `CSVFashionMNIST` dan fungsi visualisasi `save_class_grid`. Cuplikan kode krusial:
  ```python
  if not torch.cuda.is_available():
      raise RuntimeError(
          "GPU TIDAK TERDETEKSI! Eksekusi dihentikan (dilarang lanjut di CPU).\n"
          "Cara mengaktifkan GPU: Settings > Accelerator > GPU T4 x2 / GPU P100."
      )
  device = torch.device("cuda:0")

  transform_norm = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
  ])

  class CSVFashionMNIST(Dataset):
      """Dataset fallback jika internet mati: membaca CSV (label + 784 piksel) dari /kaggle/input."""
      def __init__(self, csv_file, transform=None):
          df = pd.read_csv(csv_file)
          self.labels = df.iloc[:, 0].values.astype(np.int64)
          self.images = df.iloc[:, 1:785].values.astype(np.uint8).reshape(-1, 28, 28)
          self.transform = transform

      def __getitem__(self, idx):
          img = Image.fromarray(self.images[idx], mode="L")
          if self.transform is not None:
              img = self.transform(img)
          return img, int(self.labels[idx])
  ```

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror dari Notebook**:
  ```text
  RuntimeError: DataLoader worker (pid(s) 142, 143) exited unexpectedly:
  TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  ```
- **Prompt Perbaikan ke LLM**:
  > *"Saat memproses pembuatan `TensorDataset(train_x, train_y)` dan memplot `sample_real.png`, muncul kendala konversi tensor dan lambatnya iterasi PIL Image per batch di `torchvision.datasets.FashionMNIST`. Tolong perbaiki agar `train_x` dan `test_x` dinormalisasi langsung secara vektorisasi di CPU sebagai `TensorDataset` float32 `[-1, 1]` sebelum masuk ke `DataLoader(pin_memory=True)`, sehingga `num_workers=2` bebas konflik device CUDA dan loading per epoch jauh lebih cepat."*

- **Analisis & Solusi Kelompok**:
  - **Analisis**: Menaruh tensor yang sudah dipindahkan ke `.to(device)` di dalam `Dataset` menyebabkan *worker subprocess* (`num_workers=2`) gagal melakukan *forking* CUDA context, dan transformasi `PIL -> ToTensor` di setiap epoch membuang waktu CPU sekitar ~4 detik/epoch.
  - **Solusi**: Melakukan pre-normalisasi seluruh tensor `raw_train_ds.data` di memori host (CPU) terlebih dahulu:
    ```python
    train_x = (raw_train_ds.data.float().unsqueeze(1) / 255.0 - 0.5) / 0.5  # (60000, 1, 28, 28) di CPU
    train_y = raw_train_ds.targets.long()
    train_dataset = TensorDataset(train_x, train_y)
    ```
  - **Pelajaran yang Diambil**: Dengan menyimpan `TensorDataset` yang sudah ternormalisasi di RAM CPU dan memindahkannya ke GPU di dalam loop training menggunakan `.to(device, non_blocking=True)` + `pin_memory=True`, waktu *loading* data turun drastis dan bebas dari error *multiprocessing CUDA*.

---

## Sesi 2 — Modul 2: Perancangan Arsitektur cGAN Varian A (`Embedding-Concat`) vs Varian B (`One-Hot-Concat`)

```text
LOG INTERAKSI LLM - KELOMPOK [FashionGAN]
• Tanggal Sesi: 21/10/2026
• Target Modul: Struktur Generator (Upsample + Conv2d Tanpa ConvTranspose2d) & Conditioning Spasial pada Discriminator (Varian A vs Varian B)
• Tautan Sesi Chat: [Sertakan tautan share link AI]
```

### KRONOLOGI PROMPT & RESPONS:

#### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok**:
  > *"Bangun dua varian arsitektur Conditional GAN untuk dibandingkan:*
  > *1. **Model Varian A (utama: `embedding-concat`)**:*
  > *- Generator: noise `z (100)` digabung dengan `nn.Embedding(10, 10)` menjadi `110` -> `Linear` ke `128x7x7`, `BatchNorm + ReLU` -> `Upsample(x2, nearest) + Conv2d (7->14)`, `BatchNorm + ReLU` -> `Upsample(x2, nearest) + Conv2d (14->28)`, output `Tanh`, bentuk `1x28x28`. JANGAN pakai `ConvTranspose2d` untuk menghindari artefak pola kotak-kotak.*
  > *- Discriminator: gambar `1x28x28` digabung sebagai channel dengan peta label (`nn.Embedding(10, 784)` di-reshape ke `1x28x28`) menjadi `2x28x28` -> `Conv2d` stride 2 (`2->64->128`) + `LeakyReLU(0.2)`, `Flatten`, `Linear` ke `1`, `Sigmoid`. Tanpa dropout, tanpa noise pada input, dan `BatchNorm` HANYA pada layer konvolusi kedua.*
  > *2. **Model Varian B (pembanding: `one-hot-concat`)**:*
  > *- Sama persis seperti Varian A, tetapi tanpa `nn.Embedding`: di Generator label `one-hot (10)` digabung langsung dengan `z (100)`; di Discriminator `one-hot` dibentang menjadi `10` channel konstan `28x28` lalu digabung dengan gambar (`1 + 10 = 11` channel).*
  > *Uji shape Generator dan Discriminator (A dan B) dengan tensor dummy dan tambahkan assertion bahwa `ConvTranspose2d` dan `nn.Embedding` (pada Varian B) tidak digunakan."*

- **Alasan Khusus**:
  1. Menghindari *checkerboard artifacts* (pola kisi-kisi kotak) yang umum terjadi pada DCGAN standar akibat tumpang-tindih kernel `ConvTranspose2d` dengan menggunakan `nn.Upsample(scale_factor=2, mode="nearest") + nn.Conv2d(kernel_size=3, padding=1)`.
  2. Menguji hipotesis ilmiah mengenai perbedaan ekspresivitas representasi label: apakah *spatial embedding map* terlatih `1x28x28` (`784` parameter per kelas pada `DiscriminatorA`) lebih efisien dibandingkan ekspansi *one-hot* konstan `10x28x28` (`DiscriminatorB`).

- **Output Model**:
  LLM menyusun kelas `GeneratorA`, `DiscriminatorA`, `GeneratorB`, dan `DiscriminatorB` beserta fungsi `weights_init` (`N(0, 0.02)`). Cuplikan kode krusial pada kedua Discriminator:
  ```python
  # Discriminator Varian A: Embedding(10, 784) -> (B, 1, 28, 28) -> Concat menjadi (B, 2, 28, 28)
  lbl_map = self.label_emb(labels).view(-1, 1, 28, 28)
  x = torch.cat([img, lbl_map], dim=1)  # (B, 2, 28, 28)

  # Discriminator Varian B: One-Hot (B, 10) -> Expand ke (B, 10, 28, 28) -> Concat menjadi (B, 11, 28, 28)
  one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
  lbl_map = one_hot.view(-1, self.num_classes, 1, 1).expand(-1, self.num_classes, 28, 28)
  x = torch.cat([img, lbl_map], dim=1)  # (B, 11, 28, 28)
  ```

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror dari Notebook**:
  ```text
  RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 28 but got size 1 for tensor number 1 in the list.
  ```
- **Prompt Perbaikan ke LLM**:
  > *"Pada saat menguji `DiscriminatorB` dengan tensor dummy `fake_B` berukuran `(8, 1, 28, 28)` dan label `one_hot` `(8, 10)`, `torch.cat([img, one_hot], dim=1)` menghasilkan `RuntimeError: Sizes of tensors must match except in dimension 1`. Bagaimana cara membentangkan tensor one-hot 1D menjadi peta spasial 2D `(B, 10, 28, 28)` tanpa mengalokasikan memori berlebih?"*

- **Analisis & Solusi Kelompok**:
  - **Analisis**: Tensor gambar berdimensi 4D `(B, 1, 28, 28)`, sedangkan hasil `F.one_hot(labels, 10)` masih berdimensi 2D `(B, 10)`. Penggabungan pada dimensi channel (`dim=1`) mengharuskan dimensi spasial `H=28` dan `W=28` berukuran sama.
  - **Solusi**: Mengubah dimensi *one-hot* menjadi `(B, 10, 1, 1)` menggunakan `.view(-1, 10, 1, 1)` lalu menggunakan `.expand(-1, 10, 28, 28)` (operasi *zero-copy stride broadcast* di PyTorch) sebelum `torch.cat([img, lbl_map], dim=1)`.
  - **Pelajaran yang Diambil**: `.expand()` jauh lebih hemat memori GPU dibandingkan `.repeat()` karena tidak menyalin data fisik sebanyak $28 \times 28$ kali di memori sebelum masuk ke `Conv2d(11, 64)`.

---

## Sesi 3 — Modul 3: Smoke Test, Pelatihan Adversarial 30 Epoch (Varian A & B), & Stabilitas Loss

```text
LOG INTERAKSI LLM - KELOMPOK [FashionGAN]
• Tanggal Sesi: 21/10/2026
• Target Modul: Loop Training cGAN, Inisialisasi Bobot BatchNorm, One-Sided Label Smoothing, & Checkpointing
• Tautan Sesi Chat: [Sertakan tautan share link AI]
```

### KRONOLOGI PROMPT & RESPONS:

#### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok**:
  > *"Gunakan loss `BCELoss`, optimizer `Adam(lr=2e-4, betas=(0.5, 0.999))` untuk G dan D, one-sided label smoothing (`real=0.9`, `fake=0.0`), inisialisasi bobot `N(0, 0.02)`, dan `seed=42`. Jalankan smoke test 2 epoch pada Varian A terlebih dahulu dan pastikan loss tidak `NaN` serta catat waktu per epoch. Setelah itu, latih Varian A selama 30 epoch dan Varian B selama 30 epoch dengan pengaturan yang sama persis. Simpan checkpoint (`model, optimizer, epoch, metrics`) setiap 5 epoch beserta fungsi `resume_from_checkpoint`, simpan snapshot grid 10x8 setiap 5 epoch dengan fixed noise, dan rekam metrik `loss_G, loss_D, D(x), D(G(z))` ke `outputs/metrics_A.csv` dan `metrics_B.csv`."*

- **Alasan Khusus**:
  1. *Smoke test* 2 epoch mencegah pemborosan waktu komputasi GPU jika terdapat *bug* gradien atau `NaN` di awal pelatihan.
  2. *One-sided label smoothing* (`real=0.9`) mencegah Discriminator menjadi terlalu percaya diri (*overconfident* mendekati `1.0`), sehingga gradien menuju Generator tidak menghilang (*vanishing gradient*).

- **Output Model**:
  LLM membuat fungsi modular `train_cgan_variant`, `save_checkpoint`, dan `resume_from_checkpoint`. Cuplikan kode krusial:
  ```python
  real_targets = torch.full((bsz, 1), real_smooth, device=device)  # 0.9
  fake_targets = torch.zeros((bsz, 1), device=device)              # 0.0
  gen_targets  = torch.ones((bsz, 1), device=device)               # 1.0

  # 1. Update Discriminator
  optD.zero_grad()
  out_real = netD(real_imgs, real_labels)
  loss_d_real = criterion(out_real, real_targets)

  z = torch.randn(bsz, Z_DIM, device=device)
  gen_labels = torch.randint(0, NUM_CLASSES, (bsz,), device=device)
  fake_imgs = netG(z, gen_labels)
  out_fake = netD(fake_imgs.detach(), gen_labels)
  loss_d_fake = criterion(out_fake, fake_targets)

  loss_d = loss_d_real + loss_d_fake
  loss_d.backward()
  optD.step()

  # 2. Update Generator
  optG.zero_grad()
  out_fake_g = netD(fake_imgs, gen_labels)
  loss_g = criterion(out_fake_g, gen_targets)
  loss_g.backward()
  optG.step()
  ```

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror / Gejala Bug dari Notebook**:
  ```text
  [Smoke Testpercobaan awal] Loss_D: 1.3863 | Loss_G: 0.6931 | D(x): 0.5000 | D(G(z)): 0.5000 (Gradien mati / output gambar seragam abu-abu)
  ```
- **Prompt Perbaikan ke LLM**:
  > *"Mengapa ketika fungsi `weights_init` menginisialisasi semua layer dengan `nn.init.normal_(m.weight.data, 0.0, 0.02)`, output Generator menjadi konstan nol dan gradien berhenti mengalir? Periksa bagaimana `BatchNorm2d` seharusnya diinisialisasi dan bagaimana mode `eval()` harus dipanggil saat menyimpan snapshot grid setiap 5 epoch."*

- **Analisis & Solusi Kelompok**:
  - **Analisis**: Parameter skala ($\gamma$ / `weight`) pada `nn.BatchNorm2d` merupakan pengali langsung terhadap aktivasi yang sudah dinormalisasi. Jika $\gamma$ diinisialisasi dengan rata-rata `0.0` (bukan `1.0`), sinyal aktivasi di setiap blok `BatchNorm2d` langsung teredam mendekati nol. Selain itu, saat mengambil sampel `generate_grid_array`, jika `netG` tetap dalam mode `.train()`, statistik batch dari `fixed_noise` mengubah *running mean/var* model.
  - **Solusi**: Memisahkan cabang inisialisasi `BatchNorm2d` menggunakan `nn.init.normal_(m.weight.data, 1.0, 0.02)` dan membungkus `generate_grid_array` dengan penyimpanan status `was_training = netG.training; netG.eval()` lalu mengembalikannya ke `netG.train()`.
  - **Pelajaran yang Diambil**: Inisialisasi bobot DCGAN harus membedakan antara layer bobot linier/konvolusi (`mean=0.0`) dan layer normalisasi (`mean=1.0`). Hasil *Smoke Test* langsung lulus (`7.37s/epoch`, bebas `NaN`), dan pelatihan 30 epoch berjalan stabil (`Varian A: 234.35s`, `Varian B: 254.08s`).

---

## Sesi 4 — Modul 4: Evaluasi Kuantitatif (`FashionCNN` >90%, `Fashion-FID`), & Optimasi Bertahap (`a–d`)

```text
LOG INTERAKSI LLM - KELOMPOK [FashionGAN]
• Tanggal Sesi: 22/10/2026
• Target Modul: Evaluasi Kuantitatif 10.000 Gambar Generate, Perhitungan Fréchet Distance, & Optimasi Hyperparameter
• Tautan Sesi Chat: [Sertakan tautan share link AI]
```

### KRONOLOGI PROMPT & RESPONS:

#### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok**:
  > *"Buat evaluasi kuantitatif objektif untuk membandingkan Varian A dan Varian B:*
  > *1. Latih CNN kecil (`FashionCNN`) pada data Fashion-MNIST asli sampai akurasi pada test set menembus >90%.*
  > *2. Hasilkan 1.000 gambar per kelas (total 10.000 gambar) dari masing-masing Generator (A dan B), lalu ukur akurasi CNN Classifier per kelas dan total.*
  > *3. Hitung metrik FID sederhana jika library tersedia, dan simpan tabel perbandingan ke `outputs/comparison.csv` serta visualisasi grafik berdampingan ke `outputs/comparison.png`."*

- **Alasan Khusus**:
  Kualitas model generatif tidak cukup dinilai secara subjektif dari mata manusia. Dengan melatih `FashionCNN` (`93.21%` akurasi pada data uji asli), kita dapat mengukur secara presisi apakah gambar hasil *generate* benar-benar memuat fitur morfologi pakaian sesuai label kelas yang diminta (*class-conditional accuracy*).

- **Output Model**:
  LLM mengimplementasikan `FashionCNN` dengan metode `.extract_features(x)` (vektor fitur 256-dimensi) dan fungsi `evaluate_generator_1000_per_class`.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror dari Notebook**:
  ```text
  ModuleNotFoundError: No module named 'pytorch_fid'
  ComplexWarning: Casting complex values to real discards the imaginary part (pada scipy.linalg.sqrtm matriks kovarians fitur)
  ```
- **Prompt Perbaikan ke LLM**:
  > *"Aturan proyek membatasi instalasi tambahan hanya `pip install -q gradio`, sehingga `pytorch-fid` tidak boleh diinstal. Selain itu, saat menghitung akar kuadrat perkalian matriks kovarians $\Sigma_1 \Sigma_2$ menggunakan `scipy.linalg.sqrtm`, muncul nilai imajiner kecil akibat ketidakstabilan numerik. Tolong buat fungsi `compute_frechet_distance(mu1, sigma1, mu2, sigma2)` mandiri menggunakan fitur 256-D dari `FashionCNN` yang menangani komponen imajiner dan matriks singular (`eps=1e-6`)."*

- **Analisis & Solusi Kelompok**:
  - **Analisis**: Perkalian dua matriks kovarians definit positif $\Sigma_1 \Sigma_2$ tidak selalu simetris secara numerik floating-point, sehingga `scipy.linalg.sqrtm` dapat menghasilkan bagian imajiner $\approx 10^{-12}i$.
  - **Solusi**: Menambahkan pengecekan `np.iscomplexobj(covmean)` untuk mengambil bagian `.real` serta *diagonal regularization* `offset = np.eye(sigma1.shape[0]) * eps`:
    ```python
    def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        cov_prod = sigma1 @ sigma2
        covmean, _ = linalg.sqrtm(cov_prod, disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    ```

#### 3. TAHAP OPTIMASI (Penyempurnaan Model `a–d`)
- **Prompt Optimasi ke LLM**:
  > *"Pada hasil baseline 30 epoch, akurasi kelas `Shirt (6)` (`70.50%` di A, `73.10%` di B) dan `Coat (4)` (`78.00%` di A) masih di bawah kelas lain karena kemiripan siluet atasan berlengan panjang pada resolusi 28x28. Terapkan langkah optimasi berurutan: (a) tambah epoch ke 50, (b) turunkan learning rate Discriminator menjadi `lr_D = 1e-4` (TTUR - Two Time-Scale Update Rule), (c) tambahkan layer `Conv2d(64, 32) + BatchNorm2d(32) + ReLU` pada resolusi `28x28` di Generator (`GeneratorATuned`), dan (d) evaluasi kembali pada 10.000 gambar generate."*
- **Hasil Optimasi**:
  - Akurasi CNN Classifier pada Varian A meningkat dari **`90.81%`** menjadi **`92.19%`**.
  - Nilai **Fashion-FID** membaik secara signifikan (turun dari **`3.63`** menjadi **`2.67`**, melampaui baseline Varian B `3.36`).

---

## Sesi 5 — Modul 5: Web Demo Interaktif (`app/demo.py`), Verifikasi Bit-for-Bit dengan `final_grid`, & Eksekusi Batch

```text
LOG INTERAKSI LLM - KELOMPOK [FashionGAN]
• Tanggal Sesi: 23/10/2026
• Target Modul: Pembuatan Web Demo Gradio, Debugging Reprodusibilitas RNG PyTorch (SIMD Box-Muller), & Penutupan Tunnel di Kaggle Batch
• Tautan Sesi Chat: [Sertakan tautan share link AI]
```

### KRONOLOGI PROMPT & RESPONS:

#### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok**:
  > *"Buat aplikasi web demo interaktif di `app/demo.py` menggunakan Gradio dengan: Radio pilihan varian (`Varian A` atau `Varian B`), Dropdown nama kelas dari `models/classes.json`, Slider jumlah gambar (`1–16`), input Seed opsional (`-1` = acak), tombol Generate, dan Galeri hasil yang diperbesar dari `28x28` ke `140x140` dengan interpolasi `NEAREST` agar piksel tajam. Setelah itu lakukan pengecekan wajib: pastikan generator dalam mode `eval()` dan `torch.no_grad()`, output `Tanh` dikembalikan ke `[0, 1]` lewat `(x + 1) / 2`, dan bandingkan 1 gambar dari demo (`seed=42, kelas=0`) dengan gambar pada `baris 0, kolom 0` dari `final_grid_A.png` dan `final_grid_B.png` (harus cocok identik)."*

- **Alasan Khusus**:
  Memastikan tidak ada perbedaan prapemrosesan/pascaperosesan antara kode pelatihan di notebook dan kode inferensi di aplikasi produksi (`app/demo.py`), serta menjaga ketajaman citra ukuran kecil `28x28` saat ditampilkan di layar pengguna melalui `Image.Resampling.NEAREST`.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror dari Notebook**:
  ```text
  AssertionError                            Traceback (most recent call last)
  ---> assert max_diff_A == 0 and max_diff_B == 0, "Gambar demo dan final_grid tidak identik!"
  AssertionError: Gambar demo dan final_grid tidak identik!
  (Max Pixel Diff = 187, MSE = 2415.82)
  ```
- **Prompt Perbaikan ke LLM**:
  > *"Mengapa saat kita membangkitkan noise di `final_grid` menggunakan `torch.randn((8, 100), generator=rng)` lalu mengambil sampel ke-0, hasilnya berbeda jauh dengan `torch.randn((1, 100), generator=rng)` di `app/demo.py` meskipun `seed=42` dan `torch.Generator('cpu')` yang dipakai sama persis? Bagaimana cara memperbaikinya agar `Max Pixel Diff = 0`?"*

- **Analisis & Solusi Kelompok**:
  - **Analisis**: Di PyTorch CPU, algoritma *Box-Muller* ter-vektorisasi (AVX2 SIMD) memproses bilangan acak normal dalam blok berisi **16 elemen float**. Karena dimensi laten `Z_DIM = 100` bukan kelipatan 16 ($100 = 6 \times 16 + 4$), ketika kita memanggil `torch.randn((8, 100), generator=rng)` sekaligus (800 elemen kontigu), elemen ke-100..111 diambil dari sisa buffer SIMD blok ke-7. Sebaliknya, ketika `demo.py` memanggil `torch.randn((1, 100), generator=rng)`, sisa 12 angka di buffer SIMD dibuang, atau urutan pasangan `(u1, u2)` Box-Muller bergeser!
  - **Solusi**: Membangkitkan vektor laten `z` **per sampel (1D berukuran `100`) menggunakan list comprehension + `torch.stack`** baik pada `make_fixed_noise_and_labels` di notebook maupun pada `generate_fashion_images` di `app/demo.py`:
    ```python
    rng = torch.Generator(device="cpu").manual_seed(int(seed_val))
    z = torch.stack([torch.randn(Z_DIM, generator=rng, device="cpu") for _ in range(num_images)]).to(device)
    ```
  - **Hasil Verifikasi Setelah Perbaikan**:
    ```text
    5. Perbandingan Gambar Demo vs Final Grid (Seed=42, Kelas='T-shirt/top'):
       - Varian A: Ukuran Demo = (140, 140) | Max Pixel Diff = 0 | MSE = 0.000000 -> COCOK 100% IDENTIK
       - Varian B: Ukuran Demo = (140, 140) | Max Pixel Diff = 0 | MSE = 0.000000 -> COCOK 100% IDENTIK
    ```
  - **Pelajaran yang Diambil**: Untuk menjamin determinisme lintas-ukuran-batch (*batch-size invariance*) pada generator bilangan acak PyTorch dengan dimensi bukan kelipatan 16, pembangkitan vektor laten harus dilakukan per sampel (`torch.randn(Z_DIM, generator=rng)`).

---

## Ringkasan Matriks Ketercapaian terhadap Rubrik Penilaian

| Kriteria Rubrik | Bobot | Bukti Implementasi dalam Proyek Ini |
|---|---:|---|
| **1. Kualitas & Orisinalitas Vibe Coding** | **30%** | Arsitektur bebas *checkerboard* (`Upsample + Conv2d`), komparasi ilmiah 2 metode *conditioning* (`Embedding-Concat` vs `One-Hot-Concat`), evaluator `FashionCNN` + *Fashion-FID* mandiri, serta sinkronisasi RNG deterministik antara notebook & aplikasi Gradio. |
| **2. Log Interaksi & Dokumentasi LLM** | **25%** | Rekam jejak lengkap di [`LLM_LOG.md`](LLM_LOG.md) dan [`LOG_INTERAKSI_LLM.md`](LOG_INTERAKSI_LLM.md) yang mencakup 5 sesi modul dari **Tahap Inisiasi $\rightarrow$ Penanganan Eror $\rightarrow$ Optimasi**. |
| **3. Fungsionalitas Model & Pelatihan** | **25%** | Eksekusi *Run All* tanpa error di GPU `Tesla T4`, grafik metrik valid (`loss_curve_A.png`, `loss_curve_B.png`, `comparison.png`), akurasi tinggi (`90.81%` Varian A, `92.16%` Varian B, `92.19%` Tuned), dan web demo `app/demo.py` + `demo_package.zip`. |
| **4. Kerjasama Tim & Manajemen Integrasi** | **20%** | Struktur repositori Git bersih (`.gitignore`, dokumentasi terintegrasi di `README.md`, `ANALISIS.md`, `LLM_LOG.md`, dan pembagian modul kerja yang jelas). |

