#!/usr/bin/env python3
"""Add LPIPS to Photo-SLAM ONLINE/FINAL_TAIL metrics.csv files.

This is offline evaluation only. It reads the exact GT/rendered image pairs already
used for PSNR/SSIM, computes LPIPS with the standard AlexNet backbone, and appends
an `lpips` column to each metrics.csv. Photo-SLAM tracking/mapping/optimization is
not rerun or modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    import lpips
except ImportError as e:
    raise SystemExit(
        "Missing LPIPS/PyTorch dependency in the active Python environment.\n"
        "For this Photo-SLAM setup (LibTorch 2.7.0 + CUDA 12.8), install matching Python wheels:\n"
        "  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128\n"
        "  pip install lpips pillow\n"
        f"Original import error: {e}"
    )


def image_tensor(path: Path, device: torch.device, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None and im.size != size:
            im = im.resize(size, Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def resolve_path(text: str, result_dir: Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else (result_dir / p).resolve()


def process_metrics(path: Path, result_dir: Path, model, device: torch.device) -> tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    if "left_image" not in fields or "rendered_image" not in fields:
        raise ValueError(f"Unexpected metrics schema: {path}")
    if "lpips" not in fields:
        fields.append("lpips")

    total = 0.0
    count = 0
    with torch.no_grad():
        for row in rows:
            gt_path = resolve_path(row["left_image"], result_dir)
            render_path = resolve_path(row["rendered_image"], result_dir)
            if not gt_path.exists():
                raise FileNotFoundError(f"GT image missing: {gt_path}")
            if not render_path.exists():
                raise FileNotFoundError(f"Rendered image missing: {render_path}")

            with Image.open(render_path) as pred_im:
                pred_size = pred_im.size
            pred = image_tensor(render_path, device)
            gt = image_tensor(gt_path, device, pred_size)
            value = float(model(pred, gt).reshape(-1)[0].item())
            row["lpips"] = f"{value:.10f}"
            total += value
            count += 1

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return count, (total / count if count else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = ap.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[LPIPS] backbone={args.net}, device={device}")
    model = lpips.LPIPS(net=args.net).to(device).eval()

    for subdir in ("online_tracked_view_eval", "final_tracked_view_eval"):
        path = result_dir / subdir / "metrics.csv"
        count, mean = process_metrics(path, result_dir, model, device)
        print(f"  {subdir}: {count} views, mean LPIPS={mean:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
