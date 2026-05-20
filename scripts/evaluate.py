"""
Evaluation script for MRI Super-Resolution benchmark.
Computes PSNR, SSIM metrics and generates visual comparisons.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from config import MODELS, RESULTS_DIR, TEST_HR_DIR, TEST_LR_DIR

# Configuration
TEST_HR_DIR = Path("data/test/HR")
RESULTS_DIR = Path("results")
MODELS = {}
# Auto-detect available result directories
for name in [
    "swin2sr_aligned",
    "real_esrgan",
    "swin2sr_finetuned",
    "swin2sr_postprocessed",
    "real_esrgan_postprocessed",
]:
    path = RESULTS_DIR / name
    if path.exists() and any(path.glob("*.png")):
        MODELS[name] = path



def compute_metrics(hr_dir, sr_dir):
    """Compute PSNR and SSIM between HR ground truth and SR output."""
    hr_images = sorted(hr_dir.glob("*.png"))
    psnr_values = []
    ssim_values = []

    for hr_path in hr_images:
        sr_path = sr_dir / hr_path.name
        if not sr_path.exists():
            continue

        hr_img = np.array(Image.open(hr_path).convert("L"))
        sr_img = np.array(Image.open(sr_path).convert("L"))

        # Ensure same size
        if hr_img.shape != sr_img.shape:
            sr_pil = Image.fromarray(sr_img).resize(
                (hr_img.shape[1], hr_img.shape[0]), Image.BICUBIC
            )
            sr_img = np.array(sr_pil)

        p = psnr(hr_img, sr_img, data_range=255)
        s = ssim(hr_img, sr_img, data_range=255)

        psnr_values.append(p)
        ssim_values.append(s)

    return {
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "n_images": len(psnr_values),
    }


def generate_visual_comparison(hr_dir, lr_dir, models, output_path, n_samples=5):
    """Generate a visual comparison grid of sample images."""
    hr_images = sorted(hr_dir.glob("*.png"))

    # Pick evenly spaced samples
    indices = np.linspace(0, len(hr_images) - 1, n_samples, dtype=int)
    sample_images = [hr_images[i] for i in indices]

    # Grid: n_samples rows x (LR + HR + n_models) columns
    n_cols = 2 + len(models)  # LR, HR, model outputs
    cell_size = 256
    padding = 4
    header_height = 30

    grid_w = n_cols * (cell_size + padding) + padding
    grid_h = n_samples * (cell_size + padding) + padding + header_height

    grid = Image.new("L", (grid_w, grid_h), 40)

    # Column headers
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    headers = ["LR (64x64)", "HR (256x256)"] + [f"SR ({name})" for name in models.keys()]
    for col, header in enumerate(headers):
        x = padding + col * (cell_size + padding) + cell_size // 2
        draw.text((x, 8), header, fill=220, anchor="mt")

    for row, hr_path in enumerate(sample_images):
        y = header_height + padding + row * (cell_size + padding)

        # LR image (upscaled for display)
        lr_path = lr_dir / hr_path.name
        if lr_path.exists():
            lr_img = Image.open(lr_path).convert("L").resize((cell_size, cell_size), Image.NEAREST)
            grid.paste(lr_img, (padding, y))

        # HR image
        hr_img = Image.open(hr_path).convert("L")
        grid.paste(hr_img, (padding + (cell_size + padding), y))

        # Model outputs
        for col_offset, (name, sr_dir) in enumerate(models.items()):
            sr_path = sr_dir / hr_path.name
            if sr_path.exists():
                sr_img = Image.open(sr_path).convert("L")
                if sr_img.size != (cell_size, cell_size):
                    sr_img = sr_img.resize((cell_size, cell_size), Image.BICUBIC)
                x = padding + (2 + col_offset) * (cell_size + padding)
                grid.paste(sr_img, (x, y))

    grid.save(output_path)
    print(f"Visual comparison saved to {output_path}")


def main():
    print("Evaluating super-resolution models...")
    print(f"HR ground truth: {TEST_HR_DIR}")
    print()

    # Load timings if available
    timings_path = RESULTS_DIR / "timings.json"
    timings = {}
    if timings_path.exists():
        with open(timings_path) as f:
            timings = json.load(f)

    # Compute metrics for each model
    all_metrics = {}
    for name, sr_dir in MODELS.items():
        if not sr_dir.exists():
            print(f"  Skipping {name}: no results found at {sr_dir}")
            continue

        print(f"  Evaluating {name}...")
        metrics = compute_metrics(TEST_HR_DIR, sr_dir)

        # Add timing info
        if name in timings:
            metrics["time_mean"] = timings[name]["mean"]
            metrics["time_std"] = timings[name]["std"]

        all_metrics[name] = metrics
        print(f"    PSNR: {metrics['psnr_mean']:.2f} +/- {metrics['psnr_std']:.2f} dB")
        print(f"    SSIM: {metrics['ssim_mean']:.4f} +/- {metrics['ssim_std']:.4f}")
        if "time_mean" in metrics:
            print(f"    Time: {metrics['time_mean']:.4f} +/- {metrics['time_std']:.4f} s/image")
        print()

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'Time (s/img)':<15}")
    print("=" * 70)
    for name, m in all_metrics.items():
        time_str = f"{m['time_mean']:.4f}" if "time_mean" in m else "N/A"
        print(f"{name:<15} {m['psnr_mean']:.2f} +/- {m['psnr_std']:.2f}  "
              f"{m['ssim_mean']:.4f} +/- {m['ssim_std']:.4f}  {time_str}")
    print("=" * 70)

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR / 'metrics.json'}")

    # Generate visual comparison
    generate_visual_comparison(
        TEST_HR_DIR, TEST_LR_DIR, MODELS,
        RESULTS_DIR / "visual_comparison.png",
        n_samples=5
    )


if __name__ == "__main__":
    main()
