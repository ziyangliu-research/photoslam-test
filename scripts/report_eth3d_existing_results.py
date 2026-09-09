#!/usr/bin/env python3
"""Print currently available ETH3D Photo-SLAM benchmark results.

Completed runs print their final benchmark metrics. Missing/incomplete runs are
shown as MISSING. This script never launches Photo-SLAM or modifies results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_SEQUENCES = [
    "mannequin_face_1",
    "einstein_1",
    "sofa_3",
    "plant_scene_3",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="./results")
    ap.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()

    print("================ Photo-SLAM ETH3D CURRENT RESULTS ================")
    print("Sequence          | Status  | MaxMap | Train P/S/L              | Test P/S/L               | ATE(m) | FPS   | Total(s) | Gaussians")
    print("-" * 144)

    completed = 0
    for seq in args.sequences:
        result_dir = output_root / f"eth3d_{seq}_split80_20_photoslam_final"
        summary = result_dir / "benchmark_summary.csv"
        if not summary.exists():
            print(f"{seq:17s} | MISSING | {'—':>6s} | {'—':24s} | {'—':24s} | {'—':>6s} | {'—':>5s} | {'—':>8s} | —")
            continue
        try:
            with summary.open(newline="") as f:
                row = next(csv.DictReader(f))
            tr = f"{float(row['train_psnr']):.2f}/{float(row['train_ssim']):.4f}/{float(row['train_lpips']):.4f}"
            te = f"{float(row['test_psnr']):.2f}/{float(row['test_ssim']):.4f}/{float(row['test_lpips']):.4f}"
            maxmap = 100.0 * float(row["largest_map_coverage"])
            ate = float(row["ate_rmse_m"])
            fps = float(row["fps"])
            total = float(row["total_time_sec"])
            gs = int(float(row["gaussian_count"]))
            print(
                f"{seq:17s} | OK      | {maxmap:6.2f}% | {tr:24s} | {te:24s} | "
                f"{ate:6.4f} | {fps:5.2f} | {total:8.2f} | {gs:,}"
            )
            completed += 1
        except Exception as e:
            print(f"{seq:17s} | MISSING | {'—':>6s} | {'—':24s} | {'—':24s} | {'—':>6s} | {'—':>5s} | {'—':>8s} | —")
            print(f"  incomplete/corrupt summary: {summary} ({e})")

    print("-" * 144)
    print(f"Completed: {completed}/{len(args.sequences)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
