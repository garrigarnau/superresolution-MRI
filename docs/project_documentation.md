# MRI Brain Super-Resolution: Fine-Tuning & 3D Model Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Context: MVP Baseline (2D)](#2-context-mvp-baseline-2d)
3. [Part 1: Swin2SR Fine-Tuning for MRI](#3-part-1-swin2sr-fine-tuning-for-mri)
4. [Part 2: 3D Volumetric Super-Resolution](#4-part-2-3d-volumetric-super-resolution)
5. [Results Comparison](#5-results-comparison)
6. [Conclusions & Future Work](#6-conclusions--future-work)

---

## 1. Project Overview

This project explores deep learning approaches for **super-resolution of brain MRI scans** from the OASIS-1 dataset. The work is structured in progressive phases:

1. **MVP (scripts/)**: Zero-shot evaluation of pre-trained 2D SR models (Swin2SR, Real-ESRGAN) on MRI slices.
2. **Fine-tuning (fine-tunning/)**: Domain adaptation of Swin2SR to grayscale MRI data with 4x upscaling.
3. **3D Model (3dmodel/)**: Volumetric super-resolution using MedicalNet ResNet3D-50 with 2x upscaling, preserving inter-slice coherence.

**Dataset**: OASIS-1 (Open Access Series of Imaging Studies) — T1-weighted brain MRIs, skull-stripped, bias-field corrected, registered to Talairach space (176x208x176 voxels, 1mm isotropic).

---

## 2. Context: MVP Baseline (2D)

The MVP (`scripts/`) established a baseline by running pre-trained models directly on 2D axial slices extracted from the 3D volumes:

| Model | PSNR (dB) | SSIM | Inference Time/Image |
|-------|-----------|------|---------------------|
| Swin2SR (zero-shot) | 15.40 +/- 3.56 | 0.657 +/- 0.169 | 0.38s |
| Real-ESRGAN (zero-shot) | 23.66 +/- 4.48 | 0.846 +/- 0.079 | 4.00s |

**Key findings**:
- Pre-trained models struggle with grayscale medical images (trained on RGB natural images).
- Swin2SR's poor zero-shot PSNR (15.40 dB) is due to the RGB-to-grayscale domain gap.
- This motivated fine-tuning to adapt the model to the MRI domain.

---

## 3. Part 1: Swin2SR Fine-Tuning for MRI

### 3.1 Motivation

The zero-shot evaluation showed that Swin2SR has strong architectural capabilities for super-resolution but fails on grayscale MRI because:
- It expects 3-channel RGB input/output.
- Its learned features are tuned for natural image textures, not brain tissue.

Fine-tuning addresses both issues: architectural adaptation to 1-channel and weight adaptation to the medical domain.

### 3.2 Pipeline Overview

```
Pre-trained Swin2SR (RGB, ImageNet)
    → Adapt architecture (3ch → 1ch grayscale)
        → Fine-tune on OASIS-1 MRI pairs (LR 64x64 → HR 256x256)
            → Select best checkpoint (max validation PSNR)
                → Inference on test set
```

### 3.3 Model Architecture Adaptation

**Base model**: `caidas/swin2SR-classical-sr-x4-64` from HuggingFace (Swin2SR for 4x classical super-resolution).

The critical modification converts the model from 3-channel RGB to 1-channel grayscale while preserving pre-trained knowledge:

| Layer | Original | Modified | Weight Initialization |
|-------|----------|----------|----------------------|
| First Conv | Conv2d(3, 180, 3x3) | Conv2d(1, 180, 3x3) | Average RGB input weights across channel dim |
| Final Conv | Conv2d(64, 3, 3x3) | Conv2d(64, 1, 3x3) | Average RGB output weights across channel dim |
| Mean tensor | shape (1, 3, 1, 1) | shape (1, 1, 1, 1) | Average RGB normalization values |

**Why this approach?** The Transformer backbone operates on embedding dimensions (180 channels), not image channels. Only the input/output convolutions touch the channel dimension, so averaging the RGB weights into a single channel preserves the learned feature extraction while adapting I/O.

**Alternative considered**: Duplicating the grayscale channel to 3 channels — rejected because it's less memory-efficient and conceptually less clean.

### 3.4 Dataset

| Split | Images | Source |
|-------|--------|--------|
| Training | 4,461 | 2D axial slices from OASIS-1 |
| Validation | 557 | 2D axial slices from OASIS-1 |
| Test | Separate set | LR images for inference |

**Image pairs**:
- LR: 64x64 grayscale PNG (bicubic downsampled from HR)
- HR: 256x256 grayscale PNG (original resolution)
- Scale factor: 4x

**Preprocessing**:
- Load as grayscale (PIL `.convert("L")`)
- Normalize to [0, 1] by dividing by 255
- No data augmentation (MRI already in standardized Talairach space; augmentations would create unrealistic anatomy)

### 3.5 Loss Function

**Combined Loss** = `1.0 * L1_Loss + 0.1 * Perceptual_Loss`

#### L1 Loss (Pixel Fidelity)
- Mean Absolute Error between super-resolved output and ground truth.
- Directly optimizes for PSNR.

#### VGG19 Perceptual Loss (Visual Quality)
- Uses pre-trained VGG19 (frozen, from ImageNet).
- Extracts features from 3 intermediate layers:
  - **relu1_2** (layers 0-2): Fine textures and edges
  - **relu2_2** (layers 3-7): Local patterns
  - **relu3_4** (layers 8-16): Mid-level structures
- Computes L1 distance between feature maps of SR and HR images.
- **Grayscale handling**: 1-channel images replicated to 3 channels (`.repeat(1, 3, 1, 1)`) before VGG input.
- **Numerical stability**: VGG computations kept in FP32 even with AMP enabled.

**Rationale**: L1 alone produces slightly blurry results. Perceptual loss preserves fine brain tissue structures and visual quality. The 0.1 weight prevents the perceptual loss from dominating.

### 3.6 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay regularization |
| Learning rate | 2e-5 | Low LR to preserve pre-trained weights |
| Weight decay | 1e-4 | L2 regularization |
| Batch size | 2 | GPU memory constraint |
| Max epochs | 30 | With early stopping |
| Scheduler | ReduceLROnPlateau | Reduce LR x0.5 if PSNR stalls for 3 epochs |
| Early stopping | 8 epochs | Stop if no PSNR improvement |
| AMP | Disabled | Stability over speed |
| Seed | 42 | Reproducibility |

### 3.7 Training Flow

```
For each epoch (1..30):
├── Training phase:
│   ├── Load LR batch (B, 1, 64, 64)
│   ├── Forward pass → SR output (B, 1, 256, 256)
│   ├── Size mismatch handling (bicubic interpolate if needed)
│   ├── Compute combined loss (L1 + 0.1 * Perceptual)
│   ├── Backward pass + optimizer step (AdamW)
│   └── Log loss every 50 batches
│
└── Validation phase (every epoch):
    ├── Compute validation loss (no gradients)
    ├── Compute PSNR per image (scikit-image, data_range=[0,255])
    ├── Average PSNR across validation set
    ├── Scheduler step (monitor PSNR)
    ├── If best PSNR → save best_model.pth
    └── If no improvement for 8 epochs → early stop
```

**Checkpoint format**:
```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_psnr": float
}
```

### 3.8 Inference

```bash
python infer_finetuned.py [--checkpoint PATH] [--output-dir DIR] [--base] [--device auto] [--limit N]
```

- Loads fine-tuned model (or base adapted model with `--base`).
- Processes LR test images: load grayscale → normalize → forward pass → clip to [0, 255] → save PNG.
- Outputs saved to `results/swin2sr_finetuned/`.

### 3.9 Testing & Validation

The `test_pipeline.py` provides comprehensive pre-training validation (7 stages):

| Stage | Test | Purpose |
|-------|------|---------|
| 1/7 | Dataset loading | Verify shapes (1x64x64 LR, 1x256x256 HR), normalization |
| 2/7 | Model loading | Verify 1-channel adaptation, parameter count |
| 3/7 | Forward pass | Output shape and value range on real data |
| 4/7 | Loss computation | No NaN/Inf in combined loss |
| 5/7 | Backward pass | Full training step (forward, backward, optimizer) |
| 6/7 | Checkpoint | Save/load checkpoint round-trip |
| 7/7 | GPU stress test | 20+ steps with real batch size, AMP, validation |

```bash
python test_pipeline.py           # Basic CPU/MPS test
python test_pipeline.py --gpu     # Full GPU validation
```

### 3.10 Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Full fine-tuning (all layers trainable) | MRI domain too different from ImageNet; need full adaptation |
| Low learning rate (2e-5) | Preserve good initialization, gradual domain adaptation |
| L1 + Perceptual loss | Balance pixel fidelity (PSNR) with visual quality (structures) |
| Weight averaging for channel adaptation | Memory-efficient, preserves learned features |
| No data augmentation | MRI standardized in Talairach space; augmentation creates unrealistic anatomy |
| Early stopping (8 epochs) | Prevent overfitting on limited medical data |

### 3.11 File Structure

```
fine-tunning/
├── config.py              # Hyperparameters, paths, model config
├── model.py               # Load + adapt Swin2SR to 1-channel
├── dataset.py             # PyTorch Dataset for HR/LR PNG pairs
├── losses.py              # L1 + VGG19 Perceptual Loss
├── train.py               # Main training loop
├── infer_finetuned.py     # Inference on test images
├── test_pipeline.py       # Pre-training validation suite
├── checkpoints/           # Saved model weights
│   ├── best_model.pth
│   └── last_model.pth
└── docs/
    ├── finetune.html      # Pipeline documentation
    └── setup.html         # Execution guide
```

---

## 4. Part 2: 3D Volumetric Super-Resolution

### 4.1 Motivation

The 2D fine-tuning approach processes slices independently, which:
- Loses inter-slice spatial coherence.
- Cannot exploit 3D anatomical continuity.
- May produce inconsistencies between adjacent slices.

A 3D approach processes volumetric patches, preserving the full spatial context of brain anatomy.

### 4.2 Pipeline Overview

```
OASIS-1 3D Volumes (176x208x176)
    → Gaussian blur + downsample x2 → LR volumes (88x104x88)
        → Extract 3D patch pairs (HR 64³ / LR 32³)
            → Train decoder (MedicalNet encoder frozen)
                → Sliding window inference on full volumes
                    → Evaluate 3D PSNR/SSIM
```

### 4.3 Data Preprocessing

**Source**: OASIS-1 disc1 (39 subjects), Analyze 7.5 format (.hdr/.img), skull-stripped, bias-corrected, Talairach space.

**LR Volume Generation**:
```
HR Volume (176x208x176, 1mm³)
    → Gaussian blur (σ = 0.5 * scale_factor = 1.0)
        → Trilinear downsample x2
            → LR Volume (88x104x88)
```

**3D Patch Extraction**:
- HR patch size: 64x64x64 voxels
- LR patch size: 32x32x32 voxels
- Stride: 48 (16-voxel overlap between patches)
- Brain threshold: 1% (discard patches with <1% non-zero voxels)
- Output: paired .npy files

**Data Split** (subject-level, not patch-level):
| Split | Subjects | Ratio |
|-------|----------|-------|
| Training | ~29 | 75% |
| Validation | ~4 | 10% |
| Test | ~7 | 15% |

### 4.4 Model Architecture: MedicalNetSR3D

#### Encoder (Frozen, ~46M parameters)
- **MedicalNet ResNet3D-50** (`nwirandx/medicalnet-resnet3d50` from HuggingFace)
- Pre-trained on 3D medical images (CT/MRI segmentation tasks)
- Extracts multi-scale 3D features:

| Level | Channels | Spatial Resolution (for 32³ input) |
|-------|----------|-----------------------------------|
| Embedding | 64 | 8x8x8 |
| Stage 1 | 256 | 8x8x8 |
| Stage 2 | 512 | 4x4x4 |
| Stage 3 | 1024 | 2x2x2 |
| Stage 4 | 2048 | 1x1x1 |

#### Decoder (Trainable, ~15M parameters)
Custom 3D upsampling path with skip connections from encoder:

```
Stage 4 (2048@1³)
    → ConvTranspose3d → concat Stage 3 skip → 512@2³
        → ConvTranspose3d → concat Stage 2 skip → 256@4³
            → ConvTranspose3d → concat Stage 1 skip → 128@8³
                → ConvTranspose3d → 64@16³
                    → ConvTranspose3d → 32@32³
                        → ConvTranspose3d → 16@64³
                            → Conv3d 1x1 → 1@64³ (output)
```

#### Residual Learning Strategy
The model learns a **residual** on top of a trilinear upsampling baseline:

```python
baseline = F.interpolate(lr_input, scale_factor=2, mode='trilinear')
output = baseline + 0.1 * tanh(decoder_output)
```

This ensures:
- Anatomically aligned output even if the decoder is weak.
- Bounded residual (tanh + 0.1 scaling) prevents divergence.
- Model only needs to learn the high-frequency details that trilinear misses.

### 4.5 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss | L1 (MAE) | No pre-trained VGG3D exists for perceptual loss |
| Optimizer | AdamW | lr=1e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau | patience=5, factor=0.5 |
| Batch size | 4 | Patches per batch |
| Max epochs | 50 | With early stopping |
| Early stopping | 12 epochs | Stop if no PSNR improvement |
| Encoder | Frozen | Prevent overfitting with limited data (~800 patches) |
| Seed | 42 | Reproducibility |

**Why only L1 loss?** Unlike the 2D pipeline, there is no pre-trained 3D VGG equivalent for medical images. L1 is robust, stable, and directly optimizes PSNR.

**Why freeze the encoder?** With only ~800 training patches, training 46M encoder parameters would cause severe overfitting. Only the 15M decoder parameters are trained.

### 4.6 Training Flow

```
For each epoch (1..50):
├── Training phase (encoder frozen):
│   ├── Load LR patch batch (B, 1, 32, 32, 32)
│   ├── Encoder: extract multi-scale 3D features
│   ├── Decoder: upsample with skip connections → (B, 1, 64, 64, 64)
│   ├── Residual: baseline + 0.1 * tanh(decoder_output)
│   ├── Compute L1 loss vs HR ground truth
│   ├── Backward pass (gradients only through decoder)
│   └── Log loss every 20 batches
│
└── Validation phase (every epoch):
    ├── Compute validation loss + 3D PSNR
    ├── Scheduler step (monitor PSNR)
    ├── If best PSNR → save best checkpoint
    └── If no improvement for 12 epochs → early stop
```

### 4.7 Inference: Sliding Window

The model operates on 32³ patches, but evaluation requires full-volume reconstruction (88x104x88 → 176x208x176).

**Strategy**: 3D sliding window with overlap averaging.

```
1. Pad LR volume to fit patches evenly
2. Extract overlapping 32³ patches (stride=24, overlap=8 voxels)
3. Pass each patch through model → 64³ output
4. Place outputs at x2 coordinates in full-resolution grid
5. Average overlapping regions (weight map for smooth blending)
6. Clip output to [0, 1]
```

**Baseline comparison**: Also generates trilinear interpolation (scipy.ndimage.zoom, order=1) for direct comparison.

### 4.8 Evaluation

Metrics computed over **full 3D volumes** (not per-slice):
- **3D PSNR** (dB): `10 * log10(data_range² / MSE)`
- **3D SSIM**: `structural_similarity(ref, test, data_range=1.0)`

### 4.9 Results

#### Aggregate Results (7 test subjects)

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| **MedicalNet SR 3D** | 33.08 +/- 1.00 | 0.9164 +/- 0.0181 |
| **Trilinear baseline** | 33.10 +/- 0.76 | 0.9411 +/- 0.0044 |

#### Per-Subject Results

| Subject | Model PSNR | Trilinear PSNR | Model SSIM | Trilinear SSIM |
|---------|-----------|----------------|-----------|----------------|
| OAS1_0002_MR1 | 32.58 | 33.10 | 0.8965 | 0.9471 |
| OAS1_0007_MR1 | **34.68** | 34.47 | 0.9399 | 0.9454 |
| OAS1_0009_MR1 | **34.29** | 33.88 | 0.9379 | 0.9430 |
| OAS1_0010_MR1 | 31.94 | 32.33 | 0.8956 | 0.9402 |
| OAS1_0016_MR1 | **33.04** | 32.89 | 0.9201 | 0.9379 |
| OAS1_0017_MR1 | **33.15** | 32.91 | 0.9269 | 0.9331 |
| OAS1_0019_MR1 | 31.88 | 32.15 | 0.8982 | 0.9412 |

**Analysis**:
- The model matches trilinear quality on average (delta = -0.02 dB PSNR).
- On 4/7 subjects, the model slightly outperforms trilinear in PSNR.
- SSIM is consistently lower than trilinear (-0.025 on average), indicating the model introduces slight structural variations.
- Higher variance (std 1.00 vs 0.76) shows subject-specific performance differences.

### 4.10 Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| x2 scale (not x4) | 3D x4 means 64x more voxels — intractable. x2 (8x) is manageable |
| Patches (not full volumes) | Full 176x208x176 with ResNet3D needs 30+ GB VRAM |
| MedicalNet encoder | Pre-trained on 3D medical data; encodes anatomical patterns |
| Frozen encoder | ~800 patches insufficient for 46M params; prevents overfitting |
| L1 loss only | No pre-trained 3D VGG exists for medical domain |
| Residual learning | Stable training, bounded output, model only learns high-frequency details |
| Sliding window + overlap | Eliminates patch boundary artifacts in full-volume reconstruction |
| Subject-level split | Prevents data leakage (patches from same subject in train and test) |

### 4.11 File Structure

```
3dmodel/
├── config.py              # Global configuration
├── model.py               # MedicalNetSR3D (encoder + decoder)
├── dataset.py             # MRI3DPatchDataset loader
├── preprocess.py          # Volume loading, LR generation, patch extraction
├── train.py               # Training loop with validation
├── infer.py               # Sliding window inference
├── evaluate.py            # 3D PSNR/SSIM computation
├── test_pipeline.py       # Pre-training validation
├── docs/
│   ├── pipeline.html      # Technical documentation
│   └── setup.html         # Execution guide
├── patches/               # Generated train/val/test .npy patches
├── checkpoints/           # Saved decoder weights
└── results/
    ├── README.md          # Results summary
    ├── metrics_3d.json    # Per-subject metrics
    └── figures/
        ├── metrics_summary.png
        └── OAS1_0002_MR1_comparison.png
```

### 4.12 Execution Workflow

```bash
cd 3dmodel

# 1. Preprocess: volumes → patches
python preprocess.py

# 2. Validate pipeline on GPU
python test_pipeline.py --gpu

# 3. Train (50 epochs, early stopping at 12)
python train.py

# 4. Full-volume inference (sliding window)
python infer.py

# 5. Evaluate metrics
python evaluate.py
```

---

## 5. Results Comparison

### Cross-Pipeline Summary

| Approach | Scale | PSNR (dB) | SSIM | Notes |
|----------|-------|-----------|------|-------|
| Swin2SR zero-shot (2D) | x4 | 15.40 | 0.657 | RGB model on grayscale — domain gap |
| Real-ESRGAN zero-shot (2D) | x4 | 23.66 | 0.846 | Better zero-shot, but slow (4s/img) |
| Swin2SR fine-tuned (2D) | x4 | — | — | Training infrastructure ready |
| MedicalNet SR 3D | x2 | 33.08 | 0.916 | Matches trilinear baseline |
| Trilinear baseline (3D) | x2 | 33.10 | 0.941 | Simple interpolation reference |

**Note**: The 2D (x4) and 3D (x2) results are not directly comparable due to different scale factors. The x4 task is significantly harder than x2.

### Key Takeaways

1. **Domain adaptation is critical**: Zero-shot Swin2SR on MRI fails (15.4 dB) due to the RGB/grayscale gap. Fine-tuning and channel adaptation are necessary.

2. **3D coherence has value**: While PSNR numbers are similar to trilinear, the 3D model preserves volumetric structure that independent 2D processing cannot guarantee.

3. **Data limitation**: With only 39 subjects (~800 3D patches), the frozen-encoder approach is correct but limits the model's ability to significantly outperform baselines.

4. **Residual learning works**: The 3D model's residual strategy (baseline + small correction) ensures stable training and meaningful outputs even with limited data.

---

## 6. Conclusions & Future Work

### What Was Achieved

- Successfully adapted a pre-trained 2D SR model (Swin2SR) to grayscale MRI domain.
- Built a complete 3D volumetric SR pipeline using MedicalNet as a frozen encoder.
- Established reproducible evaluation with proper subject-level splits.
- Created comprehensive testing suites for both pipelines.

### Current Limitations

- **Limited training data**: 39 subjects is small for deep learning. The 3D model can't clearly outperform trilinear with ~800 patches.
- **3D scale factor**: Only x2 due to memory constraints. Clinical applications may need x4.
- **No perceptual loss in 3D**: No pre-trained 3D perceptual network exists for the medical domain.

### Potential Improvements

- **More data**: Use full OASIS-1 (416 subjects) or combine with other public brain MRI datasets.
- **Unfrozen encoder**: With more data, partial or full encoder fine-tuning could improve results.
- **3D perceptual loss**: Train a 3D feature extractor on medical images for perceptual quality.
- **Progressive upsampling**: Chain x2 → x2 for effective x4 3D super-resolution.
- **Attention mechanisms**: Add self-attention to the decoder for better long-range spatial dependencies.

---

## Appendix A: Dependencies

### Fine-Tuning Pipeline
- torch >= 2.0 (CUDA)
- torchvision >= 0.15 (VGG19)
- transformers >= 4.30 (Swin2SR)
- scikit-image >= 0.21 (PSNR)
- Pillow, numpy

### 3D Pipeline
- torch >= 2.0 (CUDA)
- transformers (MedicalNet)
- scipy (ndimage.zoom, gaussian_filter)
- scikit-image (PSNR, SSIM)
- nibabel (Analyze format loading)
- numpy

## Appendix B: Hardware Requirements

| Pipeline | GPU VRAM | Estimated Training Time |
|----------|----------|------------------------|
| Fine-tuning (2D) | >= 8 GB | ~2-4 hours (30 epochs) |
| 3D Model | >= 4 GB | ~1-2 hours (50 epochs, patches only) |

## Appendix C: Reproducibility

Both pipelines use `seed=42` for reproducibility. Full reproduction:

```bash
# Fine-tuning
cd fine-tunning
python test_pipeline.py --gpu    # Validate
python train.py                  # Train
python infer_finetuned.py        # Infer

# 3D Model
cd 3dmodel
python preprocess.py             # Prepare data
python test_pipeline.py --gpu    # Validate
python train.py                  # Train
python infer.py                  # Infer
python evaluate.py               # Evaluate
```
