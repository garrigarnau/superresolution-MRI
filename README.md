# MRI Super-Resolution

Projecte de super-resolucio d'imatges de ressonancia magnetica cerebral (dataset OASIS).
L'objectiu es entrenar i avaluar models de super-resolucio per millorar la qualitat d'imatges MRI de baixa resolucio (64x64 -> 256x256, factor x4).

## Estructura del projecte

```
Projecte/
├── disc1/                  # Dataset OASIS original (39 subjectes)
├── data/                   # Dades preprocessades
│   ├── train/HR/           # 4461 imatges 256x256 (ground truth)
│   ├── train/LR/           # 4461 imatges 64x64 (input)
│   ├── val/HR/ val/LR/     # 557 imatges
│   └── test/HR/ test/LR/   # 559 imatges
├── scripts/
│   ├── preprocess.py       # Conversio volums 3D -> slices 2D + HR/LR pairs
│   ├── benchmark.py        # Inferencia amb Swin2SR i Real-ESRGAN
│   └── evaluate.py         # Calcul PSNR, SSIM + comparativa visual
├── results/
│   ├── swin2sr/            # Sortides del model Swin2SR
│   ├── real_esrgan/        # Sortides del model Real-ESRGAN
│   ├── metrics.json        # Metriques quantitatives
│   ├── timings.json        # Temps d'inferencia
│   └── visual_comparison.png
├── docs/
│   ├── preprocess.html     # Documentacio del preprocessament
│   └── benchmark.html      # Documentacio del benchmarking
├── weights/                # Pesos dels models (descarregats automaticament)
├── requirements.txt
└── venv/
```

## Fases del projecte

### Fase 0: Preprocessament
Conversio dels volums 3D Analyze (.hdr/.img) a parelles d'imatges 2D PNG (HR 256x256, LR 64x64).
S'utilitzen els volums T88_masked_gfc (Talairach, skull-stripped, bias-corrected).

### Fase 1: Benchmarking (sense fine-tuning)
Avaluacio de dos models pre-entrenats sobre el test set:

| Model | PSNR (dB) | SSIM | Temps (s/img) |
|-------|-----------|------|---------------|
| Swin2SR | 15.40 | 0.6574 | 0.37 |
| Real-ESRGAN | 23.66 | 0.8463 | 4.28 |

Cap dels dos models funciona be "de serie" per a imatges MRI.

### Fase 2: Fine-Tuning (pendent)

### Fase 3: Avaluacio final (pendent)

## Execucio

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Preprocessament
python scripts/preprocess.py

# Benchmarking
python scripts/benchmark.py
python scripts/evaluate.py
```

## Requisits

- Python 3.10+
- PyTorch (amb MPS/CUDA)
- nibabel, transformers, scikit-image, pillow, numpy
- Real-ESRGAN (ai-forever/Real-ESRGAN)
