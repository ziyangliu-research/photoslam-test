#!/usr/bin/env python3
"""Re-evaluate an existing FINAL_TAIL PLY using the exact pre-tail Gaussian-scene poses.

Why this exists
---------------
Photo-SLAM's tail loop calls only trainForOneIteration(); it does not consume new
mapping operations or update keyframe poses. Therefore ONLINE and FINAL_TAIL
should be compared at the same Gaussian-scene-consistent camera poses, with the
Gaussian parameters as the only changed variable.

An earlier v2 instrumentation evaluated FINAL_TAIL with shutdown-time ORB poses,
while ONLINE used the pre-tail Gaussian-scene pose snapshot. On fragmented or
pose-graph-corrected sequences those coordinate states can disagree badly.

This script does NOT rerun tracking, mapping, or optimization. It:
  1. converts online_frame_poses.csv to the checkpoint evaluator schema;
  2. loads the existing *_shutdown PLY;
  3. renders it at exactly those saved pre-tail poses;
  4. reports train/test PSNR+SSIM on the existing largest-map frame IDs.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


def read_ids(path: Path) -> set[int]:
    return {int(x.strip()) for x in path.read_text().splitlines() if x.strip()}


def find_latest_checkpoint(result_dir: Path, suffix: str) -> tuple[int, Path, Path]:
    candidates = []
    for ply in result_dir.glob(f"*_{suffix}/ply/point_cloud/iteration_*/point_cloud.ply"):
        try:
            iteration = int(ply.parent.name.split("_", 1)[1])
        except Exception:
            continue
        cameras = ply.parents[2] / "cameras.json"
        if cameras.exists():
            candidates.append((iteration, ply, cameras))
    if not candidates:
        raise FileNotFoundError(f"No *_{suffix} PLY+cameras.json found in {result_dir}")
    return max(candidates, key=lambda x: x[0])


def convert_pose_snapshot(src: Path, dst: Path, fps: float) -> int:
    count = 0
    with src.open(newline="") as f, dst.open("w", newline="") as g:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) != 17 or header[0] != "timestamp":
            raise ValueError(f"Unexpected online_frame_poses.csv schema: {header}")
        writer = csv.writer(g)
        writer.writerow(["frame_index", "timestamp"] + [f"Tcw_{r}{c}" for r in range(4) for c in range(4)])
        for row in reader:
            if len(row) != 17:
                continue
            timestamp = float(row[0])
            frame_index = int(round(timestamp * fps))
            writer.writerow([frame_index, f"{timestamp:.17g}"] + row[1:])
            count += 1
    return count


def finite_mean(rows: list[dict], key: str) -> float:
    values = []
    for r in rows:
        try:
            x = float(r[key])
        except Exception:
            continue
        if math.isfinite(x):
            values.append(x)
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--gaussian-config", default="./cfg/gaussian_mapper/Stereo/TartanAir/TartanAir_stereo_eval.yaml")
    ap.add_argument("--binary", default="./bin/tartanair_checkpoint_eval")
    ap.add_argument("--sh-degree", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    result_dir = Path(args.result_dir).resolve()
    sequence = Path(args.sequence).resolve()
    gaussian_cfg = Path(args.gaussian_config)
    if not gaussian_cfg.is_absolute():
        gaussian_cfg = (root / gaussian_cfg).resolve()
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = (root / binary).resolve()

    pose_snapshot = result_dir / "online_frame_poses.csv"
    if not pose_snapshot.exists():
        raise FileNotFoundError(f"Missing exact pose snapshot: {pose_snapshot}")

    final_iter, final_ply, final_cameras = find_latest_checkpoint(result_dir, "shutdown")
    pose_csv = result_dir / "final_tail_same_pose_eval_poses.csv"
    nposes = convert_pose_snapshot(pose_snapshot, pose_csv, args.fps)

    output_dir = result_dir / "final_tail_same_pose_eval"
    cmd = [
        str(binary), str(gaussian_cfg), str(final_ply), str(final_cameras),
        str(sequence), str(pose_csv), str(output_dir), str(args.sh_degree),
    ]
    print("[FINAL_TAIL same-pose re-evaluation]")
    print(f"  final iteration : {final_iter}")
    print(f"  pose snapshot   : {nposes} frames")
    print("  " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(root)).returncode
    if rc != 0:
        return rc

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    largest = read_ids(result_dir / "largest_map_eval" / "frame_ids.txt")
    train = read_ids(result_dir / "train_frame_ids.txt")
    test = read_ids(result_dir / "test_frame_ids.txt")

    train_rows = [r for r in rows if int(r["frame_index"]) in largest and int(r["frame_index"]) in train]
    test_rows = [r for r in rows if int(r["frame_index"]) in largest and int(r["frame_index"]) in test]

    train_psnr = finite_mean(train_rows, "psnr")
    train_ssim = finite_mean(train_rows, "ssim")
    test_psnr = finite_mean(test_rows, "psnr")
    test_ssim = finite_mean(test_rows, "ssim")

    summary = output_dir / "split_summary.txt"
    summary.write_text(
        f"final_iteration {final_iter}\n"
        f"pose_source online_frame_poses.csv\n"
        f"train_metric_frames {len(train_rows)}\n"
        f"train_psnr {train_psnr:.9f}\n"
        f"train_ssim {train_ssim:.9f}\n"
        f"test_metric_frames {len(test_rows)}\n"
        f"test_psnr {test_psnr:.9f}\n"
        f"test_ssim {test_ssim:.9f}\n"
    )

    print("\n[Same-pose FINAL_TAIL result]")
    print(f"  Train PSNR/SSIM : {train_psnr:.6f} / {train_ssim:.6f}  [{len(train_rows)} frames]")
    print(f"  Test  PSNR/SSIM : {test_psnr:.6f} / {test_ssim:.6f}  [{len(test_rows)} frames]")
    print(f"  output          : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
