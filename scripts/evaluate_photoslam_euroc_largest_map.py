#!/usr/bin/env python3
"""Largest-map + SE3/no-scale ATE evaluation for custom EuRoC Photo-SLAM runs.

No GT interpolation and no failed-pose filling are used. Camera-trajectory timestamps
are associated to the selected stride-sampled EuRoC frames, and GT poses are matched
by nearest timestamp within a small tolerance (default 10 ms), then a single SE(3)
(no scale) Kabsch alignment is applied.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

import numpy as np


def load_2d(path: Path, cols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, cols), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != cols:
        raise ValueError(f"{path}: expected {cols} columns, got {arr.shape[1]}")
    return arr


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.strip().split(maxsplit=1)
            if len(p) == 2:
                out[p[0]] = p[1]
    return out


def rigid_align_no_scale(est: np.ndarray, gt: np.ndarray):
    me, mg = est.mean(0), gt.mean(0)
    x, y = est - me, gt - mg
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = mg - r @ me
    aligned = (r @ est.T).T + t
    return aligned, r, t


def read_selected(path: Path):
    by_ts: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_ts[int(row["timestamp_ns"])] = int(row["frame_index"])
    if not by_ts:
        raise RuntimeError("selected_frames.csv is empty")
    keys = sorted(by_ts)
    return by_ts, keys


def nearest_key(keys: list[int], target: int, tol_ns: int) -> int | None:
    i = bisect.bisect_left(keys, target)
    candidates = []
    if i < len(keys): candidates.append(keys[i])
    if i > 0: candidates.append(keys[i - 1])
    if not candidates: return None
    best = min(candidates, key=lambda x: abs(x - target))
    return best if abs(best - target) <= tol_ns else None


def load_gt_csv(path: Path):
    ts, xyz = [], []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(',')
            if len(p) < 4:
                continue
            ts.append(int(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    if not ts:
        raise RuntimeError(f"No GT rows parsed from {path}")
    return np.asarray(ts, dtype=np.int64), np.asarray(xyz, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--association-tolerance-ms", type=float, default=10.0)
    args = ap.parse_args()

    result = Path(args.result_dir).resolve()
    traj = load_2d(result / "CameraTrajectory_EuRoC.txt", 8)
    selected_by_ts, selected_ts = read_selected(result / "selected_frames.csv")
    tol_ns = int(round(args.association_tolerance_ms * 1e6))

    # ORB-SLAM3's CameraTrajectory_EuRoC is already the largest Atlas map.
    rows_by_frame: dict[int, np.ndarray] = {}
    ts_by_frame: dict[int, int] = {}
    unmatched_est = 0
    for row in traj:
        est_ts = int(round(float(row[0])))
        sel_ts = nearest_key(selected_ts, est_ts, tol_ns)
        if sel_ts is None:
            unmatched_est += 1
            continue
        fid = selected_by_ts[sel_ts]
        rows_by_frame[fid] = row.copy()
        ts_by_frame[fid] = sel_ts

    frame_ids = sorted(rows_by_frame)
    if len(frame_ids) < 3:
        raise RuntimeError(f"Only {len(frame_ids)} largest-map poses match selected frames")

    tracking = read_kv(result / "tracking_summary.txt")
    input_frames = int(tracking.get("input_frames", len(selected_ts)))
    out = result / "largest_map_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frame_ids.txt").write_text("".join(f"{i}\n" for i in frame_ids))

    gt_ts, gt_xyz = load_gt_csv(Path(args.gt).resolve())
    gt_ts_list = [int(x) for x in gt_ts]
    est_xyz, matched_gt_xyz, matched_ids, matched_ts = [], [], [], []
    for fid in frame_ids:
        ts = ts_by_frame[fid]
        gt_match = nearest_key(gt_ts_list, ts, tol_ns)
        if gt_match is None:
            continue
        j = bisect.bisect_left(gt_ts_list, gt_match)
        est_xyz.append(rows_by_frame[fid][1:4])
        matched_gt_xyz.append(gt_xyz[j])
        matched_ids.append(fid)
        matched_ts.append(ts)

    if len(matched_ids) < 3:
        raise RuntimeError(f"Only {len(matched_ids)} poses associate with EuRoC GT")

    est_xyz = np.asarray(est_xyz, dtype=np.float64)
    matched_gt_xyz = np.asarray(matched_gt_xyz, dtype=np.float64)
    aligned, r, t = rigid_align_no_scale(est_xyz, matched_gt_xyz)
    err = np.linalg.norm(aligned - matched_gt_xyz, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))

    np.savetxt(out / "se3_alignment_R.txt", r, fmt="%.12f")
    np.savetxt(out / "se3_alignment_t.txt", t[None], fmt="%.12f")
    with (out / "ate_per_frame.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_ns", "translation_error_m"])
        for fid, ts, e in zip(matched_ids, matched_ts, err):
            w.writerow([fid, ts, f"{float(e):.9f}"])

    summary = [
        "selection_rule ORB_SLAM3_largest_atlas_map_by_keyframe_count",
        "dataset EuRoC",
        "stride_split_frame_id sampled_ordinal",
        "gt_association nearest_timestamp_no_interpolation",
        f"gt_association_tolerance_ms {args.association_tolerance_ms:.3f}",
        f"selected_input_frames {input_frames}",
        f"largest_map_pose_frames {len(frame_ids)}",
        f"largest_map_pose_coverage {len(frame_ids) / input_frames:.9f}",
        f"unmatched_estimated_timestamps {unmatched_est}",
        "ate_alignment SE3_no_scale",
        f"ate_matched_frames {len(matched_ids)}",
        f"ate_rmse_m {rmse:.9f}",
        f"ate_mean_m {float(np.mean(err)):.9f}",
        f"ate_median_m {float(np.median(err)):.9f}",
        f"ate_std_m {float(np.std(err)):.9f}",
    ]
    (out / "summary.txt").write_text("\n".join(summary) + "\n")

    print("[EuRoC largest-map evaluation]")
    print(f"  largest-map poses : {len(frame_ids)}/{input_frames} ({100*len(frame_ids)/input_frames:.2f}%)")
    print(f"  ATE matched       : {len(matched_ids)}")
    print(f"  ATE RMSE (SE3)    : {rmse:.6f} m")
    print(f"  output            : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
