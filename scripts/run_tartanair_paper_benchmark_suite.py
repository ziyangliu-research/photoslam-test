#!/usr/bin/env python3
"""Final Photo-SLAM paper benchmark suite for TartanAir held-out evaluation.

Default sequences: SE000 SE001 SE002 SE003.

For each sequence, one Photo-SLAM run produces:
  ONLINE      = exact pre-tail Gaussian state
  FINAL_TAIL  = original Photo-SLAM state after its unchanged tail optimization

Metrics:
  Train/Test PSNR, SSIM, LPIPS
  largest-map coverage, SE3/no-scale ATE
  ONLINE FPS
  online time, offline tail optimization time, total time
  Gaussian count

LPIPS is computed offline on the exact rendered/GT pairs already used by PSNR/SSIM.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env=None) -> int:
    print("\n>>> " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def display_float(text: str | None, fmt: str, empty: str = "—") -> str:
    if not text:
        return empty
    try:
        v = float(text)
    except ValueError:
        return empty
    if not math.isfinite(v):
        return empty
    return format(v, fmt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo")
    ap.add_argument("--gt-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt")
    ap.add_argument("--output-root", default="./results")
    ap.add_argument("--sequences", nargs="+", default=["SE000", "SE001", "SE002", "SE003"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=200)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--test-offset", type=int, default=4)
    ap.add_argument("--orb-config", default="./cfg/ORB_SLAM3/Stereo/TartanAir/TartanAirV1_Challenge.yaml")
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    gt_root = Path(args.gt_root)
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Refuse expensive runs if the local tree/binary is not the minimal paper path.
    rc = run([sys.executable, "scripts/verify_paper_eval_instrumentation.py"], root)
    if rc != 0:
        print("Paper preflight failed; refusing to start benchmark sequences.")
        return rc

    # Fail early before Photo-SLAM runs if LPIPS dependency is unavailable.
    dep = subprocess.run(
        [sys.executable, "-c", "import lpips, PIL, torch; print('LPIPS dependency OK')"],
        cwd=str(root),
    )
    if dep.returncode != 0:
        print("Install the missing evaluation dependency first: pip install lpips pillow")
        return dep.returncode

    aggregate_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for seq in args.sequences:
        sequence_dir = dataset_root / seq
        gt_path = gt_root / f"{seq}.txt"
        if args.full:
            result_name = f"tartanair_v1_{seq}_{args.start}_full_split80_20_paper_online_final"
        else:
            end = args.start + args.num_frames - 1
            result_name = f"tartanair_v1_{seq}_{args.start}_{end}_split80_20_paper_online_final"
        result_dir = output_root / result_name

        run_cmd = [
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

        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        rc = run(run_cmd, root, env=env)
        if rc != 0:
            failures.append((seq, f"Photo-SLAM run failed rc={rc}"))
            continue

        # Add LPIPS to exactly the same per-view metric rows used by PSNR/SSIM.
        lpips_cmd = [
            sys.executable,
            "scripts/add_lpips_to_photoslam_metrics.py",
            "--result-dir", str(result_dir),
            "--net", args.lpips_net,
        ]
        rc = run(lpips_cmd, root, env=env)
        if rc != 0:
            failures.append((seq, f"LPIPS evaluation failed rc={rc}"))
            continue

        # Largest coherent Atlas map + SE3/no-scale ATE.
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
            "scripts/summarize_tartanair_paper_run.py",
            "--result-dir", str(result_dir),
            "--sequence", seq,
        ]
        rc = run(sum_cmd, root)
        if rc != 0:
            failures.append((seq, f"paper summary failed rc={rc}"))
            continue

        with (result_dir / "split_benchmark_summary.csv").open(newline="") as f:
            aggregate_rows.extend(csv.DictReader(f))

    seq_tag = "_".join(args.sequences)
    if args.full:
        aggregate_name = f"tartanair_v1_{seq_tag}_{args.start}_full_split80_20_paper_summary.csv"
    else:
        end = args.start + args.num_frames - 1
        aggregate_name = f"tartanair_v1_{seq_tag}_{args.start}_{end}_split80_20_paper_summary.csv"
    aggregate_path = output_root / aggregate_name

    if aggregate_rows:
        with aggregate_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

        print("\n================ Photo-SLAM PAPER ONLINE vs FINAL_TAIL ================")
        print("Sequence | Mode       | MaxMap | Train P/S/L              | Test P/S/L               | ATE(m) | FPS   | Time O/Off/T (s)         | Gaussians")
        print("-" * 150)
        for r in aggregate_rows:
            maxmap = 100.0 * float(r["largest_map_coverage"])
            tr = f"{float(r['train_psnr']):.2f}/{float(r['train_ssim']):.4f}/{float(r['train_lpips']):.4f}"
            te = f"{float(r['test_psnr']):.2f}/{float(r['test_ssim']):.4f}/{float(r['test_lpips']):.4f}"
            ate = display_float(r.get("ate_rmse_m"), ".4f")
            fps = display_float(r.get("fps"), ".2f")
            times = (
                f"{float(r['online_time_sec']):.2f}/"
                f"{float(r['offline_opt_time_sec']):.2f}/"
                f"{float(r['total_time_sec']):.2f}"
            )
            print(
                f"{r['sequence']:7s} | {r['mode']:10s} | {maxmap:6.2f}% | "
                f"{tr:24s} | {te:24s} | {ate:>6s} | {fps:>5s} | "
                f"{times:24s} | {int(r['gaussian_count']):,}"
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
