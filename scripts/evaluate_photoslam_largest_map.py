#!/usr/bin/env python3
"""Largest-map evaluation for Photo-SLAM TartanAir runs.

This is intentionally a post-processing step: it does not change Photo-SLAM's
mapping/training behavior. ORB-SLAM3's CameraTrajectory_EuRoC.txt and
KeyFrameTrajectory_EuRoC.txt already select the Atlas map with the largest
number of keyframes. We reuse that original selection to define one coherent
map for pose and rendering evaluation.

Inputs expected under --result_dir:
  CameraTrajectory_EuRoC.txt
  KeyFrameTrajectory_EuRoC.txt
  final_tracked_view_eval/metrics.csv
  final_tracked_view_eval/rendered/*
  tracking_summary.txt

The script:
  - recovers exact dataset frame IDs belonging to ORB-SLAM3's largest map
  - reports frame count, coverage, min/max extent, and contiguous segments
  - filters the already-computed final Photo-SLAM render metrics to those views
  - creates largest_map_eval/rendered/ with relative symlinks to the original
    rendered PNGs (or copies them with --copy-rendered)
  - writes a TUM trajectory for the largest map
  - if --gt is provided, computes SE(3)-aligned translational ATE (no scale)

No failed frames are filled, and the final Gaussian scene is not modified or
split. Therefore PSNR/SSIM still reflect Photo-SLAM's actual final output while
being measured only at views belonging to its largest coherent Atlas map.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_2d(path: Path, expected_cols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, expected_cols), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != expected_cols:
        raise ValueError(
            f"{path}: expected {expected_cols} columns, got {arr.shape[1]}"
        )
    return arr


def read_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    return values


def frame_index_from_timestamp_ns(timestamp_ns: float, fps: float) -> int:
    return int(round((float(timestamp_ns) / 1e9) * fps))


def contiguous_segments(frame_ids: Iterable[int]) -> List[Tuple[int, int]]:
    ids = sorted(set(int(x) for x in frame_ids))
    if not ids:
        return []
    segments: List[Tuple[int, int]] = []
    start = prev = ids[0]
    for idx in ids[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = prev = idx
    segments.append((start, prev))
    return segments


def segments_to_text(segments: Iterable[Tuple[int, int]]) -> str:
    out = []
    for a, b in segments:
        out.append(str(a) if a == b else f"{a}-{b}")
    return ",".join(out)


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


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True,
                    help="Photo-SLAM tartanair_stereo_eval result directory")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="same synthetic timestamp rate used by the runner")
    ap.add_argument("--gt", default=None,
                    help="optional TartanAir GT txt: tx ty tz qx qy qz qw")
    ap.add_argument("--copy-rendered", action="store_true",
                    help="copy largest-map render PNGs instead of relative symlinks")
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    result_dir = Path(args.result_dir).resolve()
    cam_traj_path = result_dir / "CameraTrajectory_EuRoC.txt"
    kf_traj_path = result_dir / "KeyFrameTrajectory_EuRoC.txt"
    metrics_path = result_dir / "final_tracked_view_eval" / "metrics.csv"
    tracking_summary_path = result_dir / "tracking_summary.txt"

    for p in (cam_traj_path, kf_traj_path, metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Required result file not found: {p}")

    # ORB-SLAM3 SaveTrajectoryEuRoC() already selects the Atlas map containing
    # the largest number of KFs. Each row is:
    # timestamp_ns tx ty tz qx qy qz qw
    largest_traj = load_2d(cam_traj_path, 8)
    largest_kf_traj = load_2d(kf_traj_path, 8)

    # Deduplicate by recovered original frame index. Keep the last row if an
    # unusual duplicate timestamp appears in ORB bookkeeping.
    traj_by_frame: Dict[int, np.ndarray] = {}
    for row in largest_traj:
        frame_idx = frame_index_from_timestamp_ns(row[0], args.fps)
        traj_by_frame[frame_idx] = row.copy()

    largest_frame_ids = sorted(traj_by_frame.keys())
    if not largest_frame_ids:
        raise RuntimeError("Largest-map trajectory contains no valid frame entries")

    largest_kf_frame_ids = sorted({
        frame_index_from_timestamp_ns(row[0], args.fps)
        for row in largest_kf_traj
    })

    tracking_summary = read_key_value_file(tracking_summary_path)
    selected_input_frames = int(
        tracking_summary.get("input_frames", len(largest_frame_ids))
    )
    selected_start = int(
        tracking_summary.get("selected_start_frame", largest_frame_ids[0])
    )
    selected_end = int(
        tracking_summary.get("selected_end_frame", largest_frame_ids[-1])
    )

    eval_dir = result_dir / "largest_map_eval"
    render_dir = eval_dir / "rendered"
    eval_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    segments = contiguous_segments(largest_frame_ids)

    # Exact frame lists for common-frame comparison against other methods.
    with open(eval_dir / "frame_ids.txt", "w") as f:
        for idx in largest_frame_ids:
            f.write(f"{idx}\n")

    with open(eval_dir / "keyframe_ids.txt", "w") as f:
        for idx in largest_kf_frame_ids:
            f.write(f"{idx}\n")

    # Convert ORB-SLAM3's largest-map EuRoC trajectory to TUM timestamps in sec.
    # The pose values themselves are already camera-to-world from
    # SaveTrajectoryEuRoC().
    with open(eval_dir / "trajectory_tum.txt", "w") as f:
        for idx in largest_frame_ids:
            row = traj_by_frame[idx]
            timestamp_sec = float(row[0]) / 1e9
            f.write(
                f"{timestamp_sec:.9f} "
                + " ".join(f"{float(v):.9f}" for v in row[1:])
                + "\n"
            )

    largest_set = set(largest_frame_ids)
    filtered_rows: List[dict] = []

    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "frame_index" not in fieldnames:
            raise ValueError(f"Unexpected metrics schema: {metrics_path}")
        for row in reader:
            idx = int(row["frame_index"])
            if idx in largest_set:
                filtered_rows.append(row)

    # Keep deterministic frame order even if CSV row order changes later.
    filtered_rows.sort(key=lambda r: int(r["frame_index"]))

    with open(eval_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    rendered_frame_ids: List[int] = []
    for row in filtered_rows:
        idx = int(row["frame_index"])
        src_text = row.get("rendered_image", "")
        if not src_text:
            continue
        src = Path(src_text)
        if not src.is_absolute():
            src = (result_dir / src).resolve()
        if not src.exists():
            continue

        dst = render_dir / f"{idx:06d}.png"
        safe_unlink(dst)
        if args.copy_rendered:
            shutil.copy2(src, dst)
        else:
            rel_src = os.path.relpath(src, start=render_dir)
            dst.symlink_to(rel_src)
        rendered_frame_ids.append(idx)

    with open(eval_dir / "rendered_frame_ids.txt", "w") as f:
        for idx in rendered_frame_ids:
            f.write(f"{idx}\n")

    psnr_values = np.array(
        [float(r["psnr"]) for r in filtered_rows if r.get("psnr", "")],
        dtype=np.float64,
    )
    ssim_values = np.array(
        [float(r["ssim"]) for r in filtered_rows if r.get("ssim", "")],
        dtype=np.float64,
    )
    finite_pair_count = 0
    if len(filtered_rows):
        finite_pair_count = sum(
            np.isfinite(float(r["psnr"])) and np.isfinite(float(r["ssim"]))
            for r in filtered_rows
            if r.get("psnr", "") and r.get("ssim", "")
        )

    summary_lines = [
        "selection_rule ORB_SLAM3_largest_atlas_map_by_keyframe_count",
        "source_camera_trajectory CameraTrajectory_EuRoC.txt",
        "source_keyframe_trajectory KeyFrameTrajectory_EuRoC.txt",
        "gaussian_scene original_unsplit_final_Photo-SLAM_scene",
        f"selected_start_frame {selected_start}",
        f"selected_end_frame {selected_end}",
        f"selected_input_frames {selected_input_frames}",
        f"largest_map_pose_frames {len(largest_frame_ids)}",
        f"largest_map_pose_coverage {len(largest_frame_ids) / selected_input_frames:.9f}",
        f"largest_map_min_frame {largest_frame_ids[0]}",
        f"largest_map_max_frame {largest_frame_ids[-1]}",
        f"largest_map_contiguous_segments {segments_to_text(segments)}",
        f"largest_map_keyframes {len(largest_kf_frame_ids)}",
        f"largest_map_render_metric_frames {len(filtered_rows)}",
        f"largest_map_render_metric_coverage {len(filtered_rows) / selected_input_frames:.9f}",
        f"largest_map_saved_rendered_frames {len(rendered_frame_ids)}",
        f"finite_psnr_ssim_pairs {finite_pair_count}",
    ]

    if psnr_values.size:
        psnr_values = psnr_values[np.isfinite(psnr_values)]
        if psnr_values.size:
            summary_lines.append(f"mean_psnr {float(psnr_values.mean()):.9f}")
    if ssim_values.size:
        ssim_values = ssim_values[np.isfinite(ssim_values)]
        if ssim_values.size:
            summary_lines.append(f"mean_ssim {float(ssim_values.mean()):.9f}")

    # Optional largest-map ATE against TartanAir GT.
    if args.gt:
        gt_path = Path(args.gt).resolve()
        gt = load_2d(gt_path, 7)

        matched_ids = [idx for idx in largest_frame_ids if 0 <= idx < len(gt)]
        if len(matched_ids) < 3:
            raise RuntimeError(
                f"Only {len(matched_ids)} largest-map poses match GT; ATE needs at least 3"
            )

        est_matched = np.stack(
            [traj_by_frame[idx][1:] for idx in matched_ids], axis=0
        )
        gt_matched = gt[np.asarray(matched_ids, dtype=np.int64)]

        est_xyz = est_matched[:, :3]
        gt_xyz = gt_matched[:, :3]
        aligned_xyz, r, t = rigid_align_no_scale(est_xyz, gt_xyz)
        errors = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)

        ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
        ate_mean = float(np.mean(errors))
        ate_median = float(np.median(errors))
        ate_std = float(np.std(errors))

        np.savetxt(
            eval_dir / "ate_matched_frame_ids.txt",
            np.asarray(matched_ids, dtype=np.int64),
            fmt="%d",
        )
        np.savetxt(eval_dir / "matched_gt.txt", gt_matched, fmt="%.9f")
        np.savetxt(eval_dir / "matched_est.txt", est_matched, fmt="%.9f")
        np.savetxt(eval_dir / "se3_alignment_R.txt", r, fmt="%.12f")
        np.savetxt(eval_dir / "se3_alignment_t.txt", t[None, :], fmt="%.12f")

        with open(eval_dir / "ate_per_frame.csv", "w") as f:
            f.write("frame_index,translation_error_m\n")
            for idx, err in zip(matched_ids, errors):
                f.write(f"{idx},{float(err):.9f}\n")

        summary_lines.extend([
            "ate_alignment SE3_no_scale",
            f"ate_matched_frames {len(matched_ids)}",
            f"ate_rmse_m {ate_rmse:.9f}",
            f"ate_mean_m {ate_mean:.9f}",
            f"ate_median_m {ate_median:.9f}",
            f"ate_std_m {ate_std:.9f}",
        ])

    (eval_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("[Largest-map evaluation]")
    print(f"  selected input frames : {selected_input_frames} ({selected_start}-{selected_end})")
    print(f"  largest-map poses     : {len(largest_frame_ids)}/{selected_input_frames} "
          f"({100.0 * len(largest_frame_ids) / selected_input_frames:.2f}%)")
    print(f"  frame extent          : {largest_frame_ids[0]}-{largest_frame_ids[-1]}")
    print(f"  contiguous segments   : {segments_to_text(segments)}")
    print(f"  largest-map keyframes : {len(largest_kf_frame_ids)}")
    print(f"  render metric frames  : {len(filtered_rows)}")
    if psnr_values.size:
        print(f"  mean PSNR             : {float(psnr_values.mean()):.6f} dB")
    if ssim_values.size:
        print(f"  mean SSIM             : {float(ssim_values.mean()):.6f}")
    if args.gt:
        print(f"  ATE RMSE              : {ate_rmse:.6f} m")
    print(f"  output                : {eval_dir}")


if __name__ == "__main__":
    main()
