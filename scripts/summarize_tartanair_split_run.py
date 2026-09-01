#!/usr/bin/env python3
"""Summarize ONLINE and FINAL_TAIL Photo-SLAM results from one held-out run.

Definitions
-----------
ONLINE:
  Gaussian state at the exact boundary after the original incremental-mapping
  loop and before the original tail Gaussian optimization begins.

FINAL_TAIL:
  Original Photo-SLAM shutdown output after the unmodified tail optimization.

Both modes use the same ORB-SLAM3 trajectory / largest Atlas map. Image metrics
are evaluated on the same final-evaluable poses and are split by exact train/test
frame IDs. FPS is reported ONLY for ONLINE and uses online_pipeline_wall_sec,
which is captured before added checkpoint PLY I/O, tail optimization, and offline
metric rendering.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set


def read_kv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        parts = raw.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def read_ids(path: Path) -> List[int]:
    if not path.exists():
        return []
    return [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]


def finite_mean(rows: Iterable[dict], key: str) -> float:
    values: List[float] = []
    for row in rows:
        text = row.get(key, "")
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float, digits: int = 6) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def read_ply_vertex_count(path: Path) -> int:
    """Read the PLY vertex count from the ASCII header; payload may be binary."""
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY end_header not found: {path}")
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"PLY vertex element not found: {path}")


def find_checkpoint_ply(result_dir: Path, suffix: str) -> Path | None:
    candidates = list(
        result_dir.glob(f"*_{suffix}/ply/point_cloud/iteration_*/point_cloud.ply")
    )
    if not candidates:
        return None

    def iteration_of(p: Path) -> int:
        name = p.parent.name  # iteration_N
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return -1

    return max(candidates, key=iteration_of)


def load_metrics(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing view metrics: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_mode(
    rows: List[dict],
    train_set: Set[int],
    test_set: Set[int],
    largest_set: Set[int],
    train_selected: int,
    test_selected: int,
) -> dict:
    largest_train_pose_ids = train_set & largest_set
    largest_test_pose_ids = test_set & largest_set

    largest_train_rows = [
        r for r in rows if int(r["frame_index"]) in largest_train_pose_ids
    ]
    largest_test_rows = [
        r for r in rows if int(r["frame_index"]) in largest_test_pose_ids
    ]

    raw_train_rows = [r for r in rows if int(r["frame_index"]) in train_set]
    raw_test_rows = [r for r in rows if int(r["frame_index"]) in test_set]

    return {
        "train_metric_frames": len(largest_train_rows),
        "train_psnr": finite_mean(largest_train_rows, "psnr"),
        "train_ssim": finite_mean(largest_train_rows, "ssim"),
        "train_coverage_selected": len(largest_train_rows) / train_selected if train_selected else 0.0,
        "test_metric_frames": len(largest_test_rows),
        "test_psnr": finite_mean(largest_test_rows, "psnr"),
        "test_ssim": finite_mean(largest_test_rows, "ssim"),
        "test_coverage_selected": len(largest_test_rows) / test_selected if test_selected else 0.0,
        "raw_train_metric_frames": len(raw_train_rows),
        "raw_train_psnr": finite_mean(raw_train_rows, "psnr"),
        "raw_train_ssim": finite_mean(raw_train_rows, "ssim"),
        "raw_test_metric_frames": len(raw_test_rows),
        "raw_test_psnr": finite_mean(raw_test_rows, "psnr"),
        "raw_test_ssim": finite_mean(raw_test_rows, "ssim"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--sequence", default=None)
    args = ap.parse_args()

    result_dir = Path(args.result_dir).resolve()
    sequence = args.sequence or result_dir.name

    train_ids = read_ids(result_dir / "train_frame_ids.txt")
    test_ids = read_ids(result_dir / "test_frame_ids.txt")
    largest_ids = read_ids(result_dir / "largest_map_eval" / "frame_ids.txt")
    if not train_ids or not test_ids:
        raise FileNotFoundError("train_frame_ids.txt/test_frame_ids.txt missing or empty")
    if not largest_ids:
        raise FileNotFoundError(
            "largest_map_eval/frame_ids.txt missing. Run evaluate_photoslam_largest_map.py first."
        )

    train_set: Set[int] = set(train_ids)
    test_set: Set[int] = set(test_ids)
    largest_set: Set[int] = set(largest_ids)

    final_rows = load_metrics(result_dir / "final_tracked_view_eval" / "metrics.csv")
    online_rows = load_metrics(result_dir / "online_tracked_view_eval" / "metrics.csv")

    final_metrics = summarize_mode(
        final_rows, train_set, test_set, largest_set, len(train_ids), len(test_ids)
    )
    online_metrics = summarize_mode(
        online_rows, train_set, test_set, largest_set, len(train_ids), len(test_ids)
    )

    tracking = read_kv(result_dir / "tracking_summary.txt")
    timing = read_kv(result_dir / "timing_summary.txt")
    largest_summary = read_kv(result_dir / "largest_map_eval" / "summary.txt")

    input_frames = int(tracking.get("input_frames", len(train_ids) + len(test_ids)))
    largest_pose_frames = int(largest_summary.get("largest_map_pose_frames", len(largest_ids)))
    largest_map_coverage = (
        float(largest_summary["largest_map_pose_coverage"])
        if "largest_map_pose_coverage" in largest_summary
        else largest_pose_frames / input_frames
    )
    ate_rmse = float(largest_summary.get("ate_rmse_m", "nan"))

    online_sec = float(timing.get("online_pipeline_wall_sec", "nan"))
    online_fps = (
        input_frames / online_sec
        if math.isfinite(online_sec) and online_sec > 0
        else float("nan")
    )

    online_ply = find_checkpoint_ply(result_dir, "online")
    final_ply = find_checkpoint_ply(result_dir, "shutdown")
    if online_ply is None:
        raise FileNotFoundError("ONLINE checkpoint PLY not found")
    if final_ply is None:
        raise FileNotFoundError("FINAL_TAIL (_shutdown) PLY not found")
    online_gaussians = read_ply_vertex_count(online_ply)
    final_gaussians = read_ply_vertex_count(final_ply)

    modes = [
        ("ONLINE", online_metrics, online_sec, online_fps, online_gaussians),
        ("FINAL_TAIL", final_metrics, float("nan"), float("nan"), final_gaussians),
    ]

    summary_lines = [
        f"sequence {sequence}",
        f"input_frames {input_frames}",
        f"train_selected_frames {len(train_ids)}",
        f"test_selected_frames {len(test_ids)}",
        f"largest_map_pose_frames {largest_pose_frames}",
        f"largest_map_coverage {largest_map_coverage:.9f}",
        f"ate_rmse_m {fmt(ate_rmse, 9)}",
        "fps_policy ONLINE_only",
        "",
    ]
    for mode, metrics, sec, fps, gaussians in modes:
        prefix = mode.lower()
        summary_lines.extend([
            f"[{mode}]",
            f"{prefix}_train_metric_frames {metrics['train_metric_frames']}",
            f"{prefix}_train_metric_coverage_selected {metrics['train_coverage_selected']:.9f}",
            f"{prefix}_train_psnr {fmt(metrics['train_psnr'], 9)}",
            f"{prefix}_train_ssim {fmt(metrics['train_ssim'], 9)}",
            f"{prefix}_test_metric_frames {metrics['test_metric_frames']}",
            f"{prefix}_test_metric_coverage_selected {metrics['test_coverage_selected']:.9f}",
            f"{prefix}_test_psnr {fmt(metrics['test_psnr'], 9)}",
            f"{prefix}_test_ssim {fmt(metrics['test_ssim'], 9)}",
            f"{prefix}_pipeline_time_sec {fmt(sec, 9)}",
            f"{prefix}_fps {fmt(fps, 9)}",
            f"{prefix}_gaussian_count {gaussians}",
            "",
        ])
    (result_dir / "split_benchmark_summary.txt").write_text("\n".join(summary_lines) + "\n")

    # Two-row CSV: same trajectory fields, different Gaussian checkpoint/mode.
    csv_path = result_dir / "split_benchmark_summary.csv"
    fields = [
        "sequence", "mode", "input_frames", "train_selected", "test_selected",
        "largest_map_frames", "largest_map_coverage",
        "train_metric_frames", "train_psnr", "train_ssim", "train_coverage_selected",
        "test_metric_frames", "test_psnr", "test_ssim", "test_coverage_selected",
        "ate_rmse_m", "online_time_sec", "fps", "gaussian_count",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mode, metrics, sec, fps, gaussians in modes:
            writer.writerow({
                "sequence": sequence,
                "mode": mode,
                "input_frames": input_frames,
                "train_selected": len(train_ids),
                "test_selected": len(test_ids),
                "largest_map_frames": largest_pose_frames,
                "largest_map_coverage": fmt(largest_map_coverage, 9),
                "train_metric_frames": metrics["train_metric_frames"],
                "train_psnr": fmt(metrics["train_psnr"], 9),
                "train_ssim": fmt(metrics["train_ssim"], 9),
                "train_coverage_selected": fmt(metrics["train_coverage_selected"], 9),
                "test_metric_frames": metrics["test_metric_frames"],
                "test_psnr": fmt(metrics["test_psnr"], 9),
                "test_ssim": fmt(metrics["test_ssim"], 9),
                "test_coverage_selected": fmt(metrics["test_coverage_selected"], 9),
                "ate_rmse_m": fmt(ate_rmse, 9),
                "online_time_sec": fmt(sec, 9) if mode == "ONLINE" else "",
                "fps": fmt(fps, 9) if mode == "ONLINE" else "",
                "gaussian_count": gaussians,
            })

    print("[Photo-SLAM ONLINE vs FINAL_TAIL]")
    print(f"  sequence    : {sequence}")
    print(f"  largest map : {largest_pose_frames}/{input_frames} ({100*largest_map_coverage:.2f}%)")
    print(f"  ATE RMSE    : {fmt(ate_rmse)} m")
    for mode, metrics, sec, fps, gaussians in modes:
        print(f"  {mode}")
        print(f"    Train     : {fmt(metrics['train_psnr'])} / {fmt(metrics['train_ssim'])} "
              f"[{metrics['train_metric_frames']}/{len(train_ids)}]")
        print(f"    Test      : {fmt(metrics['test_psnr'])} / {fmt(metrics['test_ssim'])} "
              f"[{metrics['test_metric_frames']}/{len(test_ids)}]")
        if mode == "ONLINE":
            print(f"    FPS       : {fmt(fps, 3)}  (time={fmt(sec, 3)} s)")
        else:
            print("    FPS       : —  (not reported for post-sequence refinement)")
        print(f"    Gaussians : {gaussians:,}")
    print(f"  output      : {result_dir / 'split_benchmark_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
