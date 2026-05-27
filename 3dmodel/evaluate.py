"""
Evaluation for 3D MRI Super-Resolution.
Computes 3D PSNR and 3D SSIM for both model SR and trilinear baseline
against HR ground truth.
"""

import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import PATCHES_DIR, RESULTS_DIR


def compute_3d_metrics(hr_volume, sr_volume):
    """Compute PSNR and SSIM for 3D volumes."""
    # Ensure same shape
    min_shape = tuple(min(h, s) for h, s in zip(hr_volume.shape, sr_volume.shape))
    hr_crop = hr_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
    sr_crop = sr_volume[:min_shape[0], :min_shape[1], :min_shape[2]]

    sr_crop = np.clip(sr_crop, 0, 1)

    p = psnr(hr_crop, sr_crop, data_range=1.0)
    s = ssim(hr_crop, sr_crop, data_range=1.0)

    return p, s


def main():
    test_volumes_dir = PATCHES_DIR / "test_volumes"
    sr_dir = RESULTS_DIR / "model_sr"
    trilinear_dir = RESULTS_DIR / "trilinear"

    hr_files = sorted(test_volumes_dir.glob("*_hr.npy"))

    print("=" * 60)
    print("  3D SUPER-RESOLUTION EVALUATION")
    print("=" * 60)
    print(f"\nTest subjects: {len(hr_files)}")
    print()

    results = {
        "model_sr": {"psnr": [], "ssim": [], "per_subject": {}},
        "trilinear": {"psnr": [], "ssim": [], "per_subject": {}},
    }

    print(f"{'Subject':<20} {'Model PSNR':>12} {'Model SSIM':>12} "
          f"{'Trilinear PSNR':>16} {'Trilinear SSIM':>16}")
    print("-" * 80)

    for hr_path in hr_files:
        subj_id = hr_path.stem.replace("_hr", "")
        hr_volume = np.load(hr_path)

        # Model SR
        sr_path = sr_dir / f"{subj_id}_sr.npy"
        if sr_path.exists():
            sr_volume = np.load(sr_path)
            sr_psnr, sr_ssim = compute_3d_metrics(hr_volume, sr_volume)
            results["model_sr"]["psnr"].append(sr_psnr)
            results["model_sr"]["ssim"].append(sr_ssim)
            results["model_sr"]["per_subject"][subj_id] = {
                "psnr": float(sr_psnr), "ssim": float(sr_ssim)
            }
        else:
            sr_psnr, sr_ssim = None, None

        # Trilinear baseline
        tri_path = trilinear_dir / f"{subj_id}_trilinear.npy"
        if tri_path.exists():
            tri_volume = np.load(tri_path)
            tri_psnr, tri_ssim = compute_3d_metrics(hr_volume, tri_volume)
            results["trilinear"]["psnr"].append(tri_psnr)
            results["trilinear"]["ssim"].append(tri_ssim)
            results["trilinear"]["per_subject"][subj_id] = {
                "psnr": float(tri_psnr), "ssim": float(tri_ssim)
            }
        else:
            tri_psnr, tri_ssim = None, None

        sr_str = f"{sr_psnr:.2f} / {sr_ssim:.4f}" if sr_psnr else "N/A"
        tri_str = f"{tri_psnr:.2f} / {tri_ssim:.4f}" if tri_psnr else "N/A"
        print(f"{subj_id:<20} {sr_str:>24} {tri_str:>32}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for method_name, method_data in results.items():
        if method_data["psnr"]:
            mean_psnr = np.mean(method_data["psnr"])
            std_psnr = np.std(method_data["psnr"])
            mean_ssim = np.mean(method_data["ssim"])
            std_ssim = np.std(method_data["ssim"])
            method_data["psnr_mean"] = float(mean_psnr)
            method_data["psnr_std"] = float(std_psnr)
            method_data["ssim_mean"] = float(mean_ssim)
            method_data["ssim_std"] = float(std_ssim)
            method_data["n_subjects"] = len(method_data["psnr"])

            print(f"\n  {method_name}:")
            print(f"    PSNR: {mean_psnr:.2f} +/- {std_psnr:.2f} dB")
            print(f"    SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")

    # Improvement
    if results["model_sr"]["psnr"] and results["trilinear"]["psnr"]:
        psnr_improvement = results["model_sr"]["psnr_mean"] - results["trilinear"]["psnr_mean"]
        ssim_improvement = results["model_sr"]["ssim_mean"] - results["trilinear"]["ssim_mean"]
        print(f"\n  Model improvement over trilinear:")
        print(f"    PSNR: {psnr_improvement:+.2f} dB")
        print(f"    SSIM: {ssim_improvement:+.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics_3d.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Metrics saved to: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
