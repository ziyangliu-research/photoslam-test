#!/usr/bin/env python3
"""Recover ONLINE checkpoint rendering metrics from an existing dual-checkpoint run.

This does NOT rerun tracking, mapping, Gaussian optimization, or tail optimization.
It uses the already-saved ONLINE PLY/cameras.json plus ORB-SLAM3's largest-map
CameraTrajectory_EuRoC.txt.

Important: ORB-SLAM3's no-argument KeyFrameTrajectory_EuRoC export may contain
keyframes from the whole Atlas, whereas CameraTrajectory_EuRoC is selected from
the largest Atlas map. Therefore recovery deliberately DOES NOT use
KeyFrameTrajectory_EuRoC.txt. We intersect ONLINE Gaussian keyframes with the
exact largest-map frame IDs and the largest-map camera trajectory instead.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

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


def read_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]


def load_rows(path: Path, cols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != cols:
        raise ValueError(f"{path}: expected {cols} columns, got {arr.shape[1]}")
    return arr


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    q /= n
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def euroc_row_to_twc(row: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(row[4], row[5], row[6], row[7])
    T[:3, 3] = row[1:4]
    return T


def camera_json_to_twc(item: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(item["rotation"], dtype=np.float64)
    T[:3, 3] = np.asarray(item["position"], dtype=np.float64)
    return T


def frame_index_from_img_name(name: str) -> int:
    return int(Path(name).name.split("_", 1)[0])


def frame_index_from_timestamp_ns(value: float, fps: float) -> int:
    return int(round((float(value) / 1e9) * fps))


def rotation_angle_rad(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) * 0.5
    return math.acos(float(np.clip(c, -1.0, 1.0)))


def transform_residual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(a) @ b
    return float(np.linalg.norm(delta[:3, 3])), math.degrees(rotation_angle_rad(delta[:3, :3]))


def choose_consensus_transform(candidates: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, list[int]]:
    """Choose the medoid transform with the largest tight SE(3) consensus."""
    if not candidates:
        raise RuntimeError("No transform candidates")
    trans_th = 0.05
    rot_th_deg = 5.0
    best_center = 0
    best_members: list[int] = []
    for i, (_, Ti) in enumerate(candidates):
        members: list[int] = []
        for j, (_, Tj) in enumerate(candidates):
            dt, dr = transform_residual(Ti, Tj)
            if dt <= trans_th and dr <= rot_th_deg:
                members.append(j)
        if len(members) > len(best_members):
            best_center = i
            best_members = members
    return candidates[best_center][1], best_members


def append_kv(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    replaced = False
    out = []
    for line in lines:
        if line.startswith(key + " "):
            out.append(f"{key} {value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key} {value}")
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--sequence", required=True,
                    help="TartanAir sequence root containing image_left/image_right")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--gaussian-config",
                    default="./cfg/gaussian_mapper/Stereo/TartanAir/TartanAir_stereo_eval.yaml")
    ap.add_argument("--binary", default="./bin/tartanair_checkpoint_eval")
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    root = Path(__file__).resolve().parents[1]
    result_dir = Path(args.result_dir).resolve()
    sequence = Path(args.sequence).resolve()
    gaussian_cfg = (root / args.gaussian_config).resolve() if not Path(args.gaussian_config).is_absolute() else Path(args.gaussian_config)
    binary = (root / args.binary).resolve() if not Path(args.binary).is_absolute() else Path(args.binary)

    meta = read_kv(result_dir / "online_checkpoint_metadata.txt")
    if "online_iteration" not in meta:
        raise FileNotFoundError("online_checkpoint_metadata.txt / online_iteration missing")
    online_iter = int(meta["online_iteration"])
    online_sh_degree = int(meta.get("online_sh_degree", "3"))

    checkpoint_root = result_dir / f"{online_iter}_online" / "ply"
    cameras_json = checkpoint_root / "cameras.json"
    ply = checkpoint_root / "point_cloud" / f"iteration_{online_iter}" / "point_cloud.ply"
    if not cameras_json.exists():
        raise FileNotFoundError(f"Missing ONLINE cameras.json: {cameras_json}")
    if not ply.exists():
        raise FileNotFoundError(f"Missing ONLINE PLY: {ply}")

    cam_traj_path = result_dir / "CameraTrajectory_EuRoC.txt"
    largest_ids_path = result_dir / "largest_map_eval" / "frame_ids.txt"
    if not cam_traj_path.exists():
        raise FileNotFoundError("CameraTrajectory_EuRoC.txt is required for recovery")
    largest_ids = set(read_ids(largest_ids_path))
    if not largest_ids:
        raise FileNotFoundError("largest_map_eval/frame_ids.txt missing or empty")

    camera_items = json.loads(cameras_json.read_text())
    cameras_by_frame: dict[int, np.ndarray] = {}
    for item in camera_items:
        try:
            idx = frame_index_from_img_name(item["img_name"])
        except Exception:
            continue
        if idx in largest_ids:
            cameras_by_frame[idx] = camera_json_to_twc(item)

    cam_rows = load_rows(cam_traj_path, 8)
    traj_by_frame: dict[int, np.ndarray] = {}
    for row in cam_rows:
        idx = frame_index_from_timestamp_ns(row[0], args.fps)
        if idx in largest_ids:
            traj_by_frame[idx] = row

    common = sorted(set(cameras_by_frame) & set(traj_by_frame))
    if len(common) < 3:
        raise RuntimeError(
            f"Only {len(common)} largest-map Gaussian keyframes match the camera trajectory; need >=3"
        )

    candidates: list[tuple[int, np.ndarray]] = []
    for idx in common:
        Twc_export = euroc_row_to_twc(traj_by_frame[idx])
        Twc_map = cameras_by_frame[idx]
        candidates.append((idx, Twc_map @ np.linalg.inv(Twc_export)))

    T_map_from_export, consensus_members = choose_consensus_transform(candidates)
    consensus_fraction = len(consensus_members) / len(candidates)
    consensus_frames = [candidates[i][0] for i in consensus_members]

    trans_errors = []
    rot_errors_deg = []
    for i in consensus_members:
        _, Ti = candidates[i]
        dt, dr = transform_residual(T_map_from_export, Ti)
        trans_errors.append(dt)
        rot_errors_deg.append(dr)

    print(f"[Pose recovery] largest-map matched Gaussian keyframes: {len(candidates)}")
    print(f"  SE(3) consensus          : {len(consensus_members)}/{len(candidates)} ({100*consensus_fraction:.2f}%)")
    print(f"  consensus frame extent   : {min(consensus_frames)}-{max(consensus_frames)}")
    print(f"  max consensus translation residual: {max(trans_errors):.6g} m")
    print(f"  max consensus rotation residual   : {max(rot_errors_deg):.6g} deg")

    # A single rigid transform must explain nearly all matched KFs. Otherwise the
    # ONLINE checkpoint and shutdown trajectory represent genuinely different pose
    # graphs and cannot be recovered rigorously from saved artifacts alone.
    if consensus_fraction < 0.80:
        raise RuntimeError(
            "Largest-map ONLINE cameras are not related to the shutdown trajectory by one stable SE(3); "
            f"consensus is only {len(consensus_members)}/{len(candidates)} ({100*consensus_fraction:.1f}%). "
            "Refusing to fabricate ONLINE rendering poses."
        )

    pose_csv = result_dir / "online_recovered_largest_map_poses.csv"
    with pose_csv.open("w") as f:
        header = ["frame_index", "timestamp"] + [f"Tcw_{r}{c}" for r in range(4) for c in range(4)]
        f.write(",".join(header) + "\n")
        for idx in sorted(traj_by_frame):
            row = traj_by_frame[idx]
            Twc_export = euroc_row_to_twc(row)
            Twc_map = T_map_from_export @ Twc_export
            Tcw_map = np.linalg.inv(Twc_map)
            values = [str(idx), f"{idx / args.fps:.9f}"] + [f"{x:.12g}" for x in Tcw_map.reshape(-1)]
            f.write(",".join(values) + "\n")

    output_dir = result_dir / "online_tracked_view_eval"
    cmd = [
        str(binary), str(gaussian_cfg), str(ply), str(cameras_json),
        str(sequence), str(pose_csv), str(output_dir), str(online_sh_degree),
    ]
    print("[Recovered ONLINE render]\n  " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(root))
    if completed.returncode != 0:
        return completed.returncode

    timing_path = result_dir / "timing_summary.txt"
    timing = read_kv(timing_path)
    if "online_pipeline_wall_sec" not in timing and "stream_wall_sec" in timing:
        approx_sec = float(timing["stream_wall_sec"])
        input_frames = int(timing.get("input_frames", "0"))
        approx_fps = input_frames / approx_sec if approx_sec > 0 and input_frames > 0 else float("nan")
        append_kv(timing_path, "online_pipeline_wall_sec", f"{approx_sec:.9f}")
        append_kv(timing_path, "online_pipeline_fps", f"{approx_fps:.9f}")
        append_kv(timing_path, "online_timing_is_approx", "1")
        append_kv(timing_path, "online_timing_source", "stream_wall_sec")
        append_kv(timing_path, "note_online_approx_excludes_tail_but_may_omit_small_mapper_shutdown_lag", "1")

    append_kv(result_dir / "online_recovery_metadata.txt", "largest_map_matched_gaussian_keyframes", str(len(candidates)))
    append_kv(result_dir / "online_recovery_metadata.txt", "se3_consensus_keyframes", str(len(consensus_members)))
    append_kv(result_dir / "online_recovery_metadata.txt", "se3_consensus_fraction", f"{consensus_fraction:.9f}")
    append_kv(result_dir / "online_recovery_metadata.txt", "pose_source", "largest_map_CameraTrajectory_EuRoC_only")

    print(f"Recovered ONLINE metrics: {output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
