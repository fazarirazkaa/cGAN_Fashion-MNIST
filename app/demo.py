import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CLASSES_PATH = MODELS_DIR / "classes.json"
WEIGHTS_A_PATH = MODELS_DIR / "generator_A.pth"
WEIGHTS_B_PATH = MODELS_DIR / "generator_B.pth"

Z_DIM = 100
EMBED_DIM_G = 10

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


class GeneratorA(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, embed_dim=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(z_dim + embed_dim, 128 * 7 * 7, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.relu0 = nn.ReLU(True)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.relu0(self.bn0(self.fc(x).view(-1, 128, 7, 7)))
        x = self.up1(x)
        return self.up2(x)


class GeneratorB(nn.Module):
    def __init__(self, z_dim=100, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(z_dim + num_classes, 128 * 7 * 7, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.relu0 = nn.ReLU(True)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([z, one_hot], dim=1)
        x = self.relu0(self.bn0(self.fc(x).view(-1, 128, 7, 7)))
        x = self.up1(x)
        return self.up2(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG_A = GeneratorA(Z_DIM, NUM_CLASSES, EMBED_DIM_G).to(device)
netG_A.load_state_dict(torch.load(WEIGHTS_A_PATH, map_location=device))
netG_A.eval()

netG_B = GeneratorB(Z_DIM, NUM_CLASSES).to(device)
netG_B.load_state_dict(torch.load(WEIGHTS_B_PATH, map_location=device))
netG_B.eval()


def generate_fashion_images(variant_choice: str, class_name: str, num_images: int, seed_val: int):
    num_images = int(num_images)
    cls_idx = CLASS_TO_IDX.get(class_name, 0)
    model = netG_A if "Varian A" in str(variant_choice) else netG_B
    model.eval()

    if seed_val is not None and int(seed_val) >= 0:
        rng = torch.Generator(device="cpu").manual_seed(int(seed_val))
        z = torch.stack([torch.randn(Z_DIM, generator=rng, device="cpu") for _ in range(num_images)]).to(device)
    else:
        z = torch.randn((num_images, Z_DIM), device=device)

    labels = torch.full((num_images,), cls_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        fake = model(z, labels).detach().cpu()
        # Kembalikan output Tanh [-1, 1] ke [0, 1] lewat (x + 1) / 2
        fake_01 = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)

    gallery_items = []
    short_var = "Varian A" if "Varian A" in str(variant_choice) else "Varian B"
    for i in range(num_images):
        arr_28 = (fake_01[i, 0].numpy() * 255.0).round().astype(np.uint8)
        pil_28 = Image.fromarray(arr_28, mode="L")
        # Perbesar 28x28 ke 140x140 dengan interpolasi NEAREST agar tidak buram
        pil_140 = pil_28.resize((140, 140), Image.Resampling.NEAREST)
        gallery_items.append((pil_140, f"{cls_idx}: {class_name} #{i+1} ({short_var})"))

    return gallery_items


def build_demo():
    with gr.Blocks(title="Fashion-MNIST Conditional GAN Demo") as demo:
        gr.Markdown("## Web Demo Conditional GAN (cGAN) — Fashion-MNIST")
        gr.Markdown(
            "Pilih **Varian Model** (`Varian A: Embedding-Concat` vs `Varian B: One-Hot-Concat`), "
            "**Nama Kelas Pakaian**, **Jumlah Gambar (`1–16`)**, dan **Seed** (`-1` = acak). "
            "Gambar `28x28` diperbesar ke `140x140` menggunakan interpolasi `NEAREST`."
        )
        with gr.Row():
            variant_radio = gr.Radio(
                choices=["Varian A (Embedding-Concat)", "Varian B (One-Hot-Concat)"],
                value="Varian A (Embedding-Concat)",
                label="Pilihan Varian Model cGAN"
            )
            class_dropdown = gr.Dropdown(
                choices=CLASSES,
                value=CLASSES[0],
                label="Kelas Pakaian (0–9)"
            )
        with gr.Row():
            num_slider = gr.Slider(
                minimum=1, maximum=16, value=8, step=1, label="Jumlah Gambar (1–16)"
            )
            seed_input = gr.Number(
                value=42, precision=0, label="Seed Opsional (-1 = Acak)"
            )
        gen_btn = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(
            label="Galeri Hasil Generate (140x140 NEAREST)",
            columns=4,
            height="auto"
        )

        gen_btn.click(
            fn=generate_fashion_images,
            inputs=[variant_radio, class_dropdown, num_slider, seed_input],
            outputs=gallery
        )
        demo.load(
            fn=generate_fashion_images,
            inputs=[variant_radio, class_dropdown, num_slider, seed_input],
            outputs=gallery
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(share=True)
