#!/usr/bin/env python3
"""Evaluate Photo-SLAM trajectory on TartanAir V1 challenge ground truth.

The challenge GT contains one pose per input frame in:
    tx ty tz qx qy qz qw

Photo-SLAM/ORB-SLAM3 CameraTrajectory_TUM.txt contains only trajectory entries
retained by ORB-SLAM3. We recover the corresponding frame index from the
synthetic timestamp (frame_index / fps), select exactly those GT rows, and
report coverage + SE(3)-aligned ATE RMSE. No scale alignment is used because
Photo-SLAM stereo is metric-scale.

This intentionally does NOT fill failed frames with interpolation, GT, or the
last valid pose.
"""

import argparse
from pathlib import Path
import numpy as np


def load_2d(path: Path, expected_cols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != expected_cols:
        raise ValueError(f"{path}: expected {expected_cols} columns, got {arr.shape[1]}")
    return arr


def rigid_align_no_scale(est_xyz: np.ndarray, gt_xyz: np.ndarray):
    """Kabsch SE(3) alignment: aligned = R @ est + t, scale fixed to 1."""
    mu_est = est_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)

    x = est_xyz - mu_est
    y = gt_xyz - mu_gt
    h = x.T @ y
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = mu_gt - r @ mu_est
    aligned = (r @ est_xyz.T).T + t
    return aligned, r, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="TartanAir V1 GT txt: tx ty tz qx qy qz qw")
    ap.add_argument("--trajectory_tum", required=True, help="Photo-SLAM CameraTrajectory_TUM.txt")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="same synthetic timestamp rate used by tartanair_stereo_eval")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    gt_path = Path(args.gt)
    est_path = Path(args.trajectory_tum)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_2d(gt_path, 7)
    est_tum = load_2d(est_path, 8)

    # Deduplicate by recovered frame index. ORB-SLAM3 may retain repeated
    # bookkeeping timestamps around RECENTLY_LOST; using a dict makes the
    # evaluated set explicit and deterministic.
    by_frame = {}
    for row in est_tum:
        timestamp = float(row[0])
        frame_idx = int(round(timestamp * args.fps))
        if 0 <= frame_idx < len(gt):
            by_frame[frame_idx] = row.copy()

    frame_indices = np.array(sorted(by_frame.keys()), dtype=np.int64)
    if len(frame_indices) < 3:
        raise RuntimeError(f"Only {len(frame_indices)} matched poses; ATE needs at least 3")

    est_matched = np.stack([by_frame[int(i)][1:] for i in frame_indices], axis=0)
    gt_matched = gt[frame_indices]

    gt_xyz = gt_matched[:, :3]
    est_xyz = est_matched[:, :3]

    aligned_xyz, r, t = rigid_align_no_scale(est_xyz, gt_xyz)
    errors = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)

    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))
    ate_std = float(np.std(errors))
    coverage = float(len(frame_indices) / len(gt))

    np.savetxt(out_dir / "matched_frame_indices.txt", frame_indices, fmt="%d")
    np.savetxt(out_dir / "matched_gt.txt", gt_matched, fmt="%.9f")
    np.savetxt(out_dir / "matched_est.txt", est_matched, fmt="%.9f")
    np.savetxt(out_dir / "se3_alignment_R.txt", r, fmt="%.12f")
    np.savetxt(out_dir / "se3_alignment_t.txt", t[None, :], fmt="%.12f")

    with open(out_dir / "ate_per_frame.csv", "w") as f:
        f.write("frame_index,translation_error_m\n")
        for idx, err in zip(frame_indices, errors):
            f.write(f"{int(idx)},{float(err):.9f}\n")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"gt_frames {len(gt)}\n")
        f.write(f"matched_pose_frames {len(frame_indices)}\n")
        f.write(f"pose_coverage {coverage:.9f}\n")
        f.write("alignment SE3_no_scale\n")
        f.write(f"ate_rmse_m {ate_rmse:.9f}\n")
        f.write(f"ate_mean_m {ate_mean:.9f}\n")
        f.write(f"ate_median_m {ate_median:.9f}\n")
        f.write(f"ate_std_m {ate_std:.9f}\n")

    print(f"GT frames        : {len(gt)}")
    print(f"Matched poses    : {len(frame_indices)}")
    print(f"Pose coverage    : {coverage * 100.0:.2f}%")
    print(f"ATE RMSE [m]     : {ate_rmse:.6f}")
    print(f"ATE mean [m]     : {ate_mean:.6f}")
    print(f"ATE median [m]   : {ate_median:.6f}")
    print(f"Saved to         : {out_dir}")


if __name__ == "__main__":
    main()
