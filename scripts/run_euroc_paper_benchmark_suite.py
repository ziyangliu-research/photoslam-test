#!/usr/bin/env python3
"""Run Photo-SLAM paper evaluation on stride-downsampled EuRoC stereo sequences.

Default sequences: MH02 V101 V201 MH05.
Protocol:
  1) use Photo-SLAM's official EuRoC ORB/Gaussian configs + official timestamp lists;
  2) stride the original camera stream first (default stride=5);
  3) split the sampled sequence by sampled ordinal, 80/20 via every 5th sample offset 4;
  4) held-out frames keep normal stereo pose tracking but cannot initialize/insert mapping KFs;
  5) report the same paper metrics as TartanAir: P/S/L, SE3 ATE, coverage, FPS,
     online/post-sequence/total time, and Gaussian count.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

SEQUENCES = {
    "MH02": ("machine_hall/machine_hall/MH_02_easy", "MH02.txt"),
    "V101": ("vicon_room1/vicon_room1/V1_01_easy", "V101.txt"),
    "V201": ("vicon_room2/vicon_room2/V2_01_easy", "V201.txt"),
    "MH05": ("machine_hall/machine_hall/MH_05_difficult", "MH05.txt"),
}


def run(cmd: list[str], cwd: Path, env=None) -> int:
    print("\n>>> " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def show(v: str | None, fmt: str, empty="—") -> str:
    if not v:
        return empty
    try:
        x = float(v)
    except ValueError:
        return empty
    return format(x, fmt) if math.isfinite(x) else empty


def preflight(root: Path, binary: Path) -> None:
    required = [
        root / "cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml",
        root / "cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml",
        root / "ORB-SLAM3/Vocabulary/ORBvoc.txt",
        binary,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required Photo-SLAM/EuRoC resource: {p}")

    tracking_h = (root / "ORB-SLAM3/include/Tracking.h").read_text(errors="ignore")
    mapper = (root / "src/gaussian_mapper.cpp").read_text(errors="ignore")
    if "SuppressKeyFrameInsertion" not in tracking_h:
        raise RuntimeError(
            "Held-out suppression patch is missing. Run scripts/apply_tartanair_heldout_split_patch.py and rebuild ORB-SLAM3."
        )
    for token in [
        "PAPER MINIMAL ONLINE checkpoint (evaluation only)",
        "PAPER TAIL timing instrumentation (measurement only)",
    ]:
        if token not in mapper:
            raise RuntimeError(f"Paper mapper instrumentation missing: {token}")

    dep = subprocess.run(
        [sys.executable, "-c", "import torch, torchvision, lpips, PIL; print('LPIPS dependency OK')"],
        cwd=str(root),
    )
    if dep.returncode != 0:
        raise RuntimeError("Python torch/torchvision/lpips dependency is missing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/shiyo/Desktop/Datasets/EuRoC/extracted")
    ap.add_argument("--sequences", nargs="+", default=["MH02", "V101", "V201", "MH05"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--test-offset", type=int, default=4)
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--output-root", default="./results")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    binary = root / "bin/euroc_stereo_eval"
    preflight(root, binary)

    unknown = [s for s in args.sequences if s not in SEQUENCES]
    if unknown:
        raise ValueError(f"Unknown EuRoC sequence(s): {unknown}; supported={list(SEQUENCES)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    aggregate: list[dict] = []
    failures: list[tuple[str, str]] = []

    for seq in args.sequences:
        rel, ts_name = SEQUENCES[seq]
        seq_root = dataset_root / rel
        timestamps = root / "cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps" / ts_name
        gt = seq_root / "mav0/state_groundtruth_estimate0/data.csv"
        for p in [seq_root / "mav0/cam0/data", seq_root / "mav0/cam1/data", timestamps, gt]:
            if not p.exists():
                failures.append((seq, f"missing dataset/resource: {p}"))
                break
        else:
            result = output_root / f"euroc_{seq}_stride{args.stride}_split80_20_paper_online_final"
            cmd = [
                str(binary),
                str(root / "ORB-SLAM3/Vocabulary/ORBvoc.txt"),
                str(root / "cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml"),
                str(root / "cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml"),
                str(seq_root), str(timestamps), str(result),
                f"--stride={args.stride}",
                f"--test-every={args.test_every}",
                f"--test-offset={args.test_offset}",
            ]
            rc = run(cmd, root, env)
            if rc != 0:
                failures.append((seq, f"Photo-SLAM run failed rc={rc}"))
                continue

            rc = run([
                sys.executable, "scripts/add_lpips_to_photoslam_metrics.py",
                "--result-dir", str(result), "--net", args.lpips_net,
            ], root, env)
            if rc != 0:
                failures.append((seq, f"LPIPS failed rc={rc}"))
                continue

            rc = run([
                sys.executable, "scripts/evaluate_photoslam_euroc_largest_map.py",
                "--result-dir", str(result), "--gt", str(gt),
            ], root)
            if rc != 0:
                failures.append((seq, f"EuRoC largest-map/ATE evaluation failed rc={rc}"))
                continue

            rc = run([
                sys.executable, "scripts/summarize_tartanair_paper_run.py",
                "--result-dir", str(result), "--sequence", seq,
            ], root)
            if rc != 0:
                failures.append((seq, f"summary failed rc={rc}"))
                continue

            with (result / "split_benchmark_summary.csv").open(newline="") as f:
                aggregate.extend(csv.DictReader(f))

    if aggregate:
        tag = "_".join(args.sequences)
        out_csv = output_root / f"euroc_{tag}_stride{args.stride}_split80_20_paper_summary.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
            w.writeheader(); w.writerows(aggregate)

        print("\n================ Photo-SLAM EuRoC PAPER ================")
        print("Sequence | Mode       | MaxMap | Train P/S/L              | Test P/S/L               | ATE(m) | FPS   | Time O/Post/T (s)       | Gaussians")
        print("-" * 150)
        for r in aggregate:
            tr = f"{float(r['train_psnr']):.2f}/{float(r['train_ssim']):.4f}/{float(r['train_lpips']):.4f}"
            te = f"{float(r['test_psnr']):.2f}/{float(r['test_ssim']):.4f}/{float(r['test_lpips']):.4f}"
            times = f"{float(r['online_time_sec']):.2f}/{float(r['offline_opt_time_sec']):.2f}/{float(r['total_time_sec']):.2f}"
            print(
                f"{r['sequence']:7s} | {r['mode']:10s} | {100*float(r['largest_map_coverage']):6.2f}% | "
                f"{tr:24s} | {te:24s} | {show(r.get('ate_rmse_m'), '.4f'):>6s} | "
                f"{show(r.get('fps'), '.2f'):>5s} | {times:22s} | {int(r['gaussian_count']):,}"
            )
        print(f"Aggregate CSV: {out_csv}")

    if failures:
        print("\n[Failures]")
        for s, reason in failures:
            print(f"  {s}: {reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
