# Conditional GAN (cGAN) Klasik pada Fashion-MNIST (Varian A vs Varian B)

Proyek ini mengimplementasikan, melatih, dan membandingkan dua strategi *conditioning* pada **Conditional GAN (cGAN)** untuk menghasilkan gambar pakaian **Fashion-MNIST (`1x28x28`)** sesuai kelas (`0–9`):
- **Varian A (`embedding-concat`)**: Menggunakan `nn.Embedding(10, 10)` di Generator dan `nn.Embedding(10, 784)` (`1x28x28`) di Discriminator.
- **Varian B (`one-hot-concat`)**: Menggunakan vektor *one-hot* `(10)` di Generator dan peta *one-hot* konstan `10x28x28` di Discriminator tanpa `nn.Embedding`.

## Cara Menjalankan di Kaggle Notebook
1. Buka notebook `fashion_mnist_cgan.ipynb` di Kaggle Notebook.
2. Aktifkan GPU melalui menu kanan **Settings > Accelerator > GPU T4 x2 / GPU P100** dan nyalakan **Internet** (atau tambahkan dataset `zalando-research/fashionmnist` sebagai input fallback).
3. Klik **Run All** — seluruh tahapan dari pengecekan GPU, *smoke test*, training 30 epoch Varian A & B, evaluasi CNN Classifier (`10.000` gambar generate), visualisasi, verifikasi demo, hingga pembuatan `demo_package.zip` akan berjalan otomatis dari atas ke bawah tanpa error.

## Cara Menjalankan Web Demo di Laptop (CPU) — 3 Baris Instruksi
Ekstrak `demo_package.zip`, lalu jalankan **3 baris perintah** berikut di terminal:
```bash
unzip demo_package.zip -d fashion_cgan_demo && cd fashion_cgan_demo
pip install -r requirements.txt
python app/demo.py
```
