#!/usr/bin/env python3
"""Run held-out Photo-SLAM benchmarks on TartanAir SH000-SH003.

One pass per sequence produces:
  * 80% mapping/train frames + 20% held-out pose-only test frames;
  * final rendering for all final-evaluable train+test views;
  * largest-Atlas-map ATE and coverage;
  * train/test PSNR and SSIM on the largest coherent map;
  * pipeline wall time / FPS (timing excludes added offline final rendering);
  * final Gaussian count;
  * one aggregate CSV across all requested sequences.

Default mode runs 200 frames from --start. Use --full to run each sequence from
--start through its final available stereo frame.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print("\n>>> " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        parts = raw.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo")
    ap.add_argument("--gt-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt")
    ap.add_argument("--output-root", default="./results")
    ap.add_argument("--sequences", nargs="+", default=["SH000", "SH001", "SH002", "SH003"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=200)
    ap.add_argument("--full", action="store_true",
                    help="run from --start through the final available frame of each sequence")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--test-offset", type=int, default=4)
    ap.add_argument("--orb-config", default="./cfg/ORB_SLAM3/Stereo/TartanAir/TartanAirV1_Challenge.yaml")
    ap.add_argument("--cuda-device", default="0")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    gt_root = Path(args.gt_root)
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for seq in args.sequences:
        sequence_dir = dataset_root / seq
        gt_path = gt_root / f"{seq}.txt"

        if args.full:
            result_name = f"tartanair_v1_{seq}_{args.start}_full_split80_20_paper"
        else:
            end = args.start + args.num_frames - 1
            result_name = f"tartanair_v1_{seq}_{args.start}_{end}_split80_20_paper"

        result_dir = output_root / result_name

        env_prefix = ["env", f"CUDA_VISIBLE_DEVICES={args.cuda_device}"]
        run_cmd = env_prefix + [
            sys.executable,
            "scripts/run_tartanair_stereo_eval_split.py",
            "--orb-config", args.orb_config,
            "--sequence", str(sequence_dir),
            "--output", str(result_dir),
            "--start", str(args.start),
            "--fps", str(args.fps),
            "--test-every", str(args.test_every),
            "--test-offset", str(args.test_offset),
        ]
        if not args.full:
            run_cmd += ["--num-frames", str(args.num_frames)]

        rc = run(run_cmd, root)
        if rc != 0:
            failures.append((seq, f"Photo-SLAM run failed rc={rc}"))
            continue

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_photoslam_largest_map.py",
            "--result_dir", str(result_dir),
            "--gt", str(gt_path),
            "--fps", str(args.fps),
        ]
        rc = run(eval_cmd, root)
        if rc != 0:
            failures.append((seq, f"largest-map evaluation failed rc={rc}"))
            continue

        sum_cmd = [
            sys.executable,
            "scripts/summarize_tartanair_split_run.py",
            "--result-dir", str(result_dir),
            "--sequence", seq,
        ]
        rc = run(sum_cmd, root)
        if rc != 0:
            failures.append((seq, f"split summary failed rc={rc}"))
            continue

        one_row = result_dir / "split_benchmark_summary.csv"
        with one_row.open(newline="") as f:
            aggregate_rows.extend(csv.DictReader(f))

    if args.full:
        aggregate_name = f"tartanair_v1_SH000_SH003_{args.start}_full_split80_20_summary.csv"
    else:
        end = args.start + args.num_frames - 1
        aggregate_name = f"tartanair_v1_SH000_SH003_{args.start}_{end}_split80_20_summary.csv"
    aggregate_path = output_root / aggregate_name

    if aggregate_rows:
        fieldnames = list(aggregate_rows[0].keys())
        with aggregate_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_rows)

        print("\n================ Aggregate benchmark ================")
        print("Sequence | MaxMap | Train PSNR/SSIM | Test PSNR/SSIM | ATE(m) | FPS | Gaussians")
        for r in aggregate_rows:
            print(
                f"{r['sequence']:7s} | "
                f"{100*float(r['largest_map_coverage']):6.2f}% | "
                f"{float(r['train_psnr']):6.2f}/{float(r['train_ssim']):.4f} | "
                f"{float(r['test_psnr']):6.2f}/{float(r['test_ssim']):.4f} | "
                f"{float(r['ate_rmse_m']):.4f} | "
                f"{float(r['fps']):6.2f} | "
                f"{int(r['gaussian_count']):,}"
            )
        print(f"Aggregate CSV: {aggregate_path}")

    if failures:
        print("\n[Failures]")
        for seq, reason in failures:
            print(f"  {seq}: {reason}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
