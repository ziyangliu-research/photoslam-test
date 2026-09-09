#!/usr/bin/env python3
"""Summarize one final-only Photo-SLAM ETH3D rectified stereo run.

Protocol:
  - strict 8:2 split on input frame order (every fifth frame, offset 4, is test)
  - test frames track pose but are suppressed from persistent mapping/KF insertion
  - one official final Photo-SLAM map is evaluated (no ONLINE/FINAL_TAIL split)
  - PSNR/SSIM/LPIPS are reported on the largest ORB-SLAM3 Atlas map
  - ATE is SE(3)-aligned, no scale, against rectified-left ETH3D ground truth
  - reported FPS = input frames / full Photo-SLAM method wall time
  - full method wall time ends when GaussianMapper exits and excludes our added
    final metric rendering, LPIPS, ATE, and summary post-processing
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        parts = raw.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def read_ids(path: Path) -> set[int]:
    return {int(x.strip()) for x in path.read_text().splitlines() if x.strip()}


def finite_mean(rows: Iterable[dict], key: str) -> float:
    vals = []
    for r in rows:
        try:
            v = float(r.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise RuntimeError(f"PLY vertex count not found: {path}")


def find_final_ply(result_dir: Path) -> Path:
    candidates = list(result_dir.glob("*_shutdown/ply/point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError("Final Photo-SLAM _shutdown PLY not found")
    def iteration(p: Path) -> int:
        try:
            return int(p.parent.name.split("_", 1)[1])
        except Exception:
            return -1
    return max(candidates, key=iteration)


def rigid_align_no_scale(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    mu_e = est_xyz.mean(axis=0)
    mu_g = gt_xyz.mean(axis=0)
    x = est_xyz - mu_e
    y = gt_xyz - mu_g
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = mu_g - r @ mu_e
    return (r @ est_xyz.T).T + t


def interpolate_gt_translation(gt: np.ndarray, ts: float, max_span: float) -> np.ndarray | None:
    times = gt[:, 0]
    pos = np.searchsorted(times, ts)
    if pos < len(times) and abs(times[pos] - ts) < 1e-12:
        return gt[pos, 1:4].copy()
    if pos == 0 or pos >= len(times):
        return None
    t0, t1 = times[pos - 1], times[pos]
    if (t1 - t0) <= 0 or (t1 - t0) > max_span:
        return None
    a = (ts - t0) / (t1 - t0)
    if a < -1e-9 or a > 1.0 + 1e-9:
        return None
    return (1.0 - a) * gt[pos - 1, 1:4] + a * gt[pos, 1:4]


def safe_link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, start=dst.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--max-gt-interpolation-span", type=float, default=0.05)
    args = ap.parse_args()

    result_dir = Path(args.result_dir).resolve()
    gt_path = Path(args.groundtruth).resolve()
    metrics_path = result_dir / "final_tracked_view_eval" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(newline="") as f:
        metrics = list(csv.DictReader(f))
    if metrics and "lpips" not in metrics[0]:
        raise RuntimeError("LPIPS column missing; run add_lpips_to_photoslam_metrics.py first")

    train_ids = read_ids(result_dir / "train_frame_ids.txt")
    test_ids = read_ids(result_dir / "test_frame_ids.txt")
    tracking = read_kv(result_dir / "tracking_summary.txt")
    timing = read_kv(result_dir / "timing_summary.txt")

    traj = np.loadtxt(result_dir / "CameraTrajectory_EuRoC.txt", dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[None, :]
    if traj.shape[1] != 8:
        raise ValueError("CameraTrajectory_EuRoC.txt must have 8 columns")
    largest_ts_ns = {int(round(float(row[0]))) for row in traj}

    def in_largest(r: dict) -> bool:
        return int(r["timestamp_ns"]) in largest_ts_ns

    train_rows = [r for r in metrics if int(r["frame_index"]) in train_ids and in_largest(r)]
    test_rows = [r for r in metrics if int(r["frame_index"]) in test_ids and in_largest(r)]

    input_frames = int(tracking.get("input_frames", len(train_ids) + len(test_ids)))
    largest_frames = len(traj)
    coverage = largest_frames / input_frames if input_frames else 0.0
    strict_success_rate = float(tracking.get("strict_success_rate", "nan"))

    # ETH3D GT is timestamp tx ty tz qx qy qz qw, timestamp in seconds.
    gt = np.loadtxt(gt_path, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt[None, :]
    if gt.shape[1] != 8:
        raise ValueError(f"Expected 8-column ETH3D GT: {gt_path}")
    gt = gt[np.argsort(gt[:, 0])]

    est_xyz = []
    gt_xyz = []
    ate_timestamps = []
    for row in traj:
        ts = float(row[0]) / 1e9
        g = interpolate_gt_translation(gt, ts, args.max_gt_interpolation_span)
        if g is None:
            continue
        est_xyz.append(row[1:4])
        gt_xyz.append(g)
        ate_timestamps.append(ts)
    if len(est_xyz) < 3:
        raise RuntimeError(f"Only {len(est_xyz)} trajectory poses matched ETH3D GT")
    est_xyz_arr = np.asarray(est_xyz, dtype=np.float64)
    gt_xyz_arr = np.asarray(gt_xyz, dtype=np.float64)
    aligned = rigid_align_no_scale(est_xyz_arr, gt_xyz_arr)
    errors = np.linalg.norm(aligned - gt_xyz_arr, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))

    total_sec = float(timing.get("pipeline_until_gaussian_mapper_exit_wall_sec", "nan"))
    fps = input_frames / total_sec if math.isfinite(total_sec) and total_sec > 0 else float("nan")
    gaussians = read_ply_vertex_count(find_final_ply(result_dir))

    # Explicit test-view visualization folder requested for paper inspection.
    test_render_dir = result_dir / "test_rendered"
    test_gt_dir = result_dir / "test_gt"
    test_render_dir.mkdir(exist_ok=True)
    test_gt_dir.mkdir(exist_ok=True)
    for r in test_rows:
        idx = int(r["frame_index"])
        render = Path(r["rendered_image"])
        gt_img = Path(r["left_image"])
        if not render.is_absolute(): render = (result_dir / render).resolve()
        if not gt_img.is_absolute(): gt_img = (result_dir / gt_img).resolve()
        if render.exists(): safe_link(render, test_render_dir / f"{idx:06d}.png")
        if gt_img.exists(): safe_link(gt_img, test_gt_dir / f"{idx:06d}.png")

    row = {
        "sequence": args.sequence,
        "input_frames": input_frames,
        "train_selected": len(train_ids),
        "test_selected": len(test_ids),
        "largest_map_frames": largest_frames,
        "largest_map_coverage": coverage,
        "strict_success_rate": strict_success_rate,
        "train_metric_frames": len(train_rows),
        "train_psnr": finite_mean(train_rows, "psnr"),
        "train_ssim": finite_mean(train_rows, "ssim"),
        "train_lpips": finite_mean(train_rows, "lpips"),
        "test_metric_frames": len(test_rows),
        "test_psnr": finite_mean(test_rows, "psnr"),
        "test_ssim": finite_mean(test_rows, "ssim"),
        "test_lpips": finite_mean(test_rows, "lpips"),
        "ate_rmse_m": ate_rmse,
        "ate_matched_frames": len(errors),
        "fps": fps,
        "total_time_sec": total_sec,
        "gaussian_count": gaussians,
    }

    fields = list(row.keys())
    with (result_dir / "benchmark_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerow(row)

    def fnum(x: float, n: int) -> str:
        return "nan" if not math.isfinite(float(x)) else f"{float(x):.{n}f}"

    lines = [
        f"sequence {args.sequence}",
        f"input_frames {input_frames}",
        f"train_selected {len(train_ids)}",
        f"test_selected {len(test_ids)}",
        f"largest_map_frames {largest_frames}",
        f"largest_map_coverage {coverage:.9f}",
        f"strict_success_rate {strict_success_rate:.9f}",
        f"train_psnr {fnum(row['train_psnr'], 9)}",
        f"train_ssim {fnum(row['train_ssim'], 9)}",
        f"train_lpips {fnum(row['train_lpips'], 9)}",
        f"test_psnr {fnum(row['test_psnr'], 9)}",
        f"test_ssim {fnum(row['test_ssim'], 9)}",
        f"test_lpips {fnum(row['test_lpips'], 9)}",
        "ate_alignment SE3_no_scale",
        "gt_timestamp_policy linear_translation_interpolation_between_bracketing_ETH3D_GT_poses",
        f"ate_matched_frames {len(errors)}",
        f"ate_rmse_m {ate_rmse:.9f}",
        f"fps_full_official_pipeline {fnum(fps, 9)}",
        f"total_time_sec {fnum(total_sec, 9)}",
        "time_policy stream_start_to_GaussianMapper_exit_excluding_added_final_metrics_LPIPS_ATE",
        f"gaussian_count {gaussians}",
        f"test_render_dir {test_render_dir}",
    ]
    (result_dir / "benchmark_summary.txt").write_text("\n".join(lines) + "\n")

    with (result_dir / "ate_per_frame.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["timestamp", "translation_error_m"])
        w.writerows(zip(ate_timestamps, errors.tolist()))

    print(f"[{args.sequence}] Photo-SLAM final-only")
    print(f"  coverage      : {largest_frames}/{input_frames} ({coverage*100:.2f}%)")
    print(f"  Train P/S/L   : {row['train_psnr']:.3f} / {row['train_ssim']:.4f} / {row['train_lpips']:.4f}")
    print(f"  Test  P/S/L   : {row['test_psnr']:.3f} / {row['test_ssim']:.4f} / {row['test_lpips']:.4f}")
    print(f"  ATE SE3       : {ate_rmse:.4f} m")
    print(f"  FPS           : {fps:.3f}")
    print(f"  Total time    : {total_sec:.3f} s")
    print(f"  Gaussians     : {gaussians:,}")
    print(f"  Test renders  : {test_render_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
