# MRI Super-Resolution

Super-resolution project for OASIS brain MRI slices. The task is 4x
single-image super-resolution from 64x64 LR images to 256x256 HR images.

## Current Organization

There is one canonical Swin2SR baseline in this repository: `swin2sr`.
It uses the aligned behavior. The old unaligned Swin2SR output is not kept as a
separate final MVP method.

The root `results/` folder is reserved for the no-fine-tuning MVP benchmark:

```text
results/
├── real_esrgan/
├── swin2sr/
├── metrics.json
├── timings.json
└── visual_comparison.png
```

The fine-tuning comparison lives inside the fine-tuning folder:

```text
fine-tunning/results/
├── real_esrgan/
├── swin2sr/
├── swin2sr_finetuned/
├── metrics.json
├── timings.json
└── visual_comparison.png
```

`fine-tunning/` is the existing project folder name. It is kept to avoid
breaking local paths.

## Project Structure

```text
.
├── data/                         # Preprocessed train/val/test PNG pairs
├── disc1/                        # Original OASIS volumes
├── docs/
│   └── PIPELINE_PROFESSOR.md     # Pipeline and result organization notes
├── fine-tunning/
│   ├── checkpoints/              # Fine-tuned Swin2SR checkpoints, if present
│   ├── infer_finetuned.py
│   ├── train.py
│   └── results/                  # Global/fine-tuning comparison results
├── results/                      # No-fine-tuning MVP benchmark only
├── scripts/
│   ├── benchmark.py              # Pretrained Swin2SR + Real-ESRGAN inference
│   ├── evaluate.py               # Metrics and visual comparisons
│   ├── align_swin2sr.py          # Legacy repair script only
│   └── swin2sr_alignment.py      # Canonical Swin2SR alignment helper
└── weights/
```

## Results

No-fine-tuning MVP benchmark in `results/`:

| Method | PSNR (dB) | SSIM | Time (s/img) |
|---|---:|---:|---:|
| Swin2SR aligned pretrained | 22.41 +/- 9.01 | 0.7619 +/- 0.2278 | 0.1083 |
| Real-ESRGAN pretrained | 23.66 +/- 4.48 | 0.8463 +/- 0.0793 | 0.6360 |

Fine-tuning/global comparison in `fine-tunning/results/`:

| Method | PSNR (dB) | SSIM | Time (s/img) |
|---|---:|---:|---:|
| Swin2SR aligned pretrained | 22.41 +/- 9.01 | 0.7619 +/- 0.2278 | 0.1083 |
| Real-ESRGAN pretrained | 23.66 +/- 4.48 | 0.8463 +/- 0.0793 | 0.6360 |
| Swin2SR fine-tuned | 37.04 +/- 2.69 | 0.9764 +/- 0.0131 | N/A |

The fine-tuned timing is not reported because the existing fine-tuned inference
outputs did not include a timing JSON entry.

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Preprocess OASIS volumes into HR/LR PNG pairs.
python scripts/preprocess.py

# No-fine-tuning MVP benchmark.
python scripts/benchmark.py --device auto
python scripts/evaluate.py

# Fine-tuned Swin2SR inference, if checkpoints exist.
python fine-tunning/infer_finetuned.py --device auto

# Global comparison including fine-tuning outputs.
python scripts/evaluate.py --results-dir fine-tunning/results
```

On Windows PowerShell, activate the environment with:

```powershell
.\venv\Scripts\Activate.ps1
```

## Notes

- `scripts/benchmark.py` writes aligned Swin2SR outputs directly to
  `results/swin2sr/`.
- `scripts/align_swin2sr.py` is kept only as a compatibility script for older
  unaligned output folders.
- Do not add `results/swin2sr_aligned/` as a final method. If alignment is
  needed, it belongs in the canonical `swin2sr` path.
