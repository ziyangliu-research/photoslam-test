#!/usr/bin/env python3
"""Recover ONLINE metrics for existing SH000-SH003 full-sequence runs.

No SLAM/mapping/optimization is rerun. This script only:
  1. renders each already-saved *_online checkpoint on recovered largest-map poses;
  2. reruns the lightweight summary step;
  3. aggregates ONLINE + FINAL_TAIL rows into one CSV.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="+", default=["SH000", "SH001", "SH002", "SH003"])
    ap.add_argument("--dataset-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo")
    ap.add_argument("--gt-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt")
    ap.add_argument("--output-root", default="./results")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    gt_root = Path(args.gt_root)

    rows: list[dict] = []
    for seq in args.sequences:
        result_dir = output_root / f"tartanair_v1_{seq}_0_full_split80_20_online_final"
        if not result_dir.exists():
            raise FileNotFoundError(f"Missing result directory: {result_dir}")

        largest_summary = result_dir / "largest_map_eval" / "summary.txt"
        if not largest_summary.exists():
            run([
                sys.executable, "scripts/evaluate_photoslam_largest_map.py",
                "--result_dir", str(result_dir),
                "--gt", str(gt_root / f"{seq}.txt"),
                "--fps", str(args.fps),
            ], root)

        run([
            sys.executable, "scripts/recover_online_checkpoint_eval.py",
            "--result-dir", str(result_dir),
            "--sequence", str(dataset_root / seq),
            "--fps", str(args.fps),
        ], root)

        run([
            sys.executable, "scripts/summarize_tartanair_split_run.py",
            "--result-dir", str(result_dir),
            "--sequence", seq,
        ], root)

        with (result_dir / "split_benchmark_summary.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))

    aggregate = output_root / "tartanair_v1_SH000_SH003_0_full_split80_20_online_final_recovered_summary.csv"
    if rows:
        with aggregate.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n================ Recovered ONLINE / FINAL_TAIL benchmark ================")
    print("Sequence | Mode       | MaxMap | Train PSNR/SSIM | Test PSNR/SSIM | ATE(m) | FPS* | Gaussians")
    for r in rows:
        fps = "—" if r["mode"] != "ONLINE" or not r.get("fps") else f"{float(r['fps']):.2f}"
        print(
            f"{r['sequence']:7s} | {r['mode']:10s} | "
            f"{100*float(r['largest_map_coverage']):6.2f}% | "
            f"{float(r['train_psnr']):6.2f}/{float(r['train_ssim']):.4f} | "
            f"{float(r['test_psnr']):6.2f}/{float(r['test_ssim']):.4f} | "
            f"{float(r['ate_rmse_m']):.4f} | {fps:>5s} | {int(r['gaussian_count']):,}"
        )
    print("* For the affected historical runs, ONLINE FPS uses stream_wall_sec as an explicitly marked approximation.")
    print(f"Aggregate CSV: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
