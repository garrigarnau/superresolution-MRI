"""Run fine-tuned Swin2SR inference on MRI test images."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CHECKPOINT_DIR, MODEL_NAME, PROJECT_ROOT
from model import load_swin2sr_grayscale


TEST_LR_DIR = PROJECT_ROOT / "data" / "test" / "LR"
FINE_TUNING_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_DIR = FINE_TUNING_RESULTS_DIR / "swin2sr_finetuned"


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def load_inference_model(checkpoint_path, device):
    model = load_swin2sr_grayscale(MODEL_NAME)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_lr_image(path, device):
    image = Image.open(path).convert("L")
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def save_sr_image(tensor, path):
    image = tensor.squeeze().detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    output = Image.fromarray(image, mode="L")
    if output.size != (256, 256):
        output = output.resize((256, 256), Image.BICUBIC)
    output.save(path)


def save_timings(timings, device):
    timings_path = FINE_TUNING_RESULTS_DIR / "timings.json"
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    if timings_path.exists():
        with open(timings_path) as f:
            timings_data = json.load(f)
    else:
        timings_data = {}

    timings_data["device"] = str(device)
    timings_data["n_images"] = len(timings)
    timings_data["swin2sr_finetuned"] = {
        "mean": float(np.mean(timings)),
        "std": float(np.std(timings)),
        "total": float(np.sum(timings)),
        "per_image": [float(t) for t in timings],
    }

    with open(timings_path, "w") as f:
        json.dump(timings_data, f, indent=2)

    print(f"Timings saved to {timings_path}")


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuned Swin2SR on test images.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best_model.pth",
        help="Checkpoint to load. Defaults to fine-tunning/checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where generated SR images will be saved.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test images to process for a quick preview.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run train.py first."
        )

    device = resolve_device(args.device)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test LR images: {TEST_LR_DIR}")
    print(f"Output: {args.output_dir}")
    print()

    model, checkpoint = load_inference_model(args.checkpoint, device)
    if checkpoint and "epoch" in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if checkpoint and "best_psnr" in checkpoint:
        print(f"Best validation PSNR: {checkpoint['best_psnr']:.2f} dB")

    image_paths = sorted(TEST_LR_DIR.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    print(f"Running inference on {len(image_paths)} images...")
    timings = []
    with torch.no_grad():
        for index, image_path in enumerate(image_paths, 1):
            lr_tensor = load_lr_image(image_path, device)

            start = time.time()
            sr_tensor = model(lr_tensor).reconstruction
            elapsed = time.time() - start
            timings.append(elapsed)

            save_sr_image(sr_tensor, args.output_dir / image_path.name)

            if index == 1 or index % 50 == 0 or index == len(image_paths):
                print(f"  [{index}/{len(image_paths)}] avg {np.mean(timings):.4f}s/image")

    print()
    print(f"Saved {len(image_paths)} images to {args.output_dir}")
    save_timings(timings, device)


if __name__ == "__main__":
    main()
