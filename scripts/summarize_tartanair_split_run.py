#!/usr/bin/env python3
"""Summarize one held-out Photo-SLAM TartanAir run for paper tables.

Expected run protocol:
  - selected frames: typically 0..199 (200 total)
  - held-out split: every 5 frames, offset 4 -> 160 train / 40 test
  - test frames estimate pose but cannot initialize/insert ORB/Gaussian KFs
  - offline final rendering covers all final-evaluable frames

Formal image metrics are reported on the largest ORB-SLAM3 Atlas map, split by
exact train/test frame IDs. ATE is read from evaluate_photoslam_largest_map.py
(SE3 alignment, no scale). Timing uses pipeline-to-GaussianMapper-exit, which is
recorded before the added offline final-view rendering.
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

    metrics_path = result_dir / "final_tracked_view_eval" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing final-view metrics: {metrics_path}")

    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    train_set: Set[int] = set(train_ids)
    test_set: Set[int] = set(test_ids)
    largest_set: Set[int] = set(largest_ids)

    # Only frames in the coherent largest Atlas map are formal image metrics.
    largest_train_pose_ids = train_set & largest_set
    largest_test_pose_ids = test_set & largest_set

    largest_train_rows = [
        r for r in rows
        if int(r["frame_index"]) in largest_train_pose_ids
    ]
    largest_test_rows = [
        r for r in rows
        if int(r["frame_index"]) in largest_test_pose_ids
    ]

    # Also retain raw split metrics as diagnostics; formal table uses largest-map.
    raw_train_rows = [r for r in rows if int(r["frame_index"]) in train_set]
    raw_test_rows = [r for r in rows if int(r["frame_index"]) in test_set]

    tracking = read_kv(result_dir / "tracking_summary.txt")
    timing = read_kv(result_dir / "timing_summary.txt")
    largest_summary = read_kv(result_dir / "largest_map_eval" / "summary.txt")
    gauss = read_kv(result_dir / "final_gaussian_count.txt")

    input_frames = int(tracking.get("input_frames", len(train_ids) + len(test_ids)))
    pipeline_sec = float(timing.get("pipeline_until_gaussian_mapper_exit_wall_sec", "nan"))
    fps = input_frames / pipeline_sec if math.isfinite(pipeline_sec) and pipeline_sec > 0 else float("nan")

    largest_pose_frames = int(largest_summary.get("largest_map_pose_frames", len(largest_ids)))
    largest_map_coverage = (
        float(largest_summary["largest_map_pose_coverage"])
        if "largest_map_pose_coverage" in largest_summary
        else largest_pose_frames / input_frames
    )
    ate_rmse = float(largest_summary.get("ate_rmse_m", "nan"))
    gaussian_count = int(gauss.get("final_gaussian_count", tracking.get("final_gaussian_count", "0")))

    train_psnr = finite_mean(largest_train_rows, "psnr")
    train_ssim = finite_mean(largest_train_rows, "ssim")
    test_psnr = finite_mean(largest_test_rows, "psnr")
    test_ssim = finite_mean(largest_test_rows, "ssim")

    train_metric_frames = len(largest_train_rows)
    test_metric_frames = len(largest_test_rows)
    train_largest_pose_frames = len(largest_train_pose_ids)
    test_largest_pose_frames = len(largest_test_pose_ids)

    train_cov_selected = train_metric_frames / len(train_ids)
    test_cov_selected = test_metric_frames / len(test_ids)
    train_cov_largest = (
        train_metric_frames / train_largest_pose_frames if train_largest_pose_frames else 0.0
    )
    test_cov_largest = (
        test_metric_frames / test_largest_pose_frames if test_largest_pose_frames else 0.0
    )

    summary_lines = [
        f"sequence {sequence}",
        f"input_frames {input_frames}",
        f"train_selected_frames {len(train_ids)}",
        f"test_selected_frames {len(test_ids)}",
        f"largest_map_pose_frames {largest_pose_frames}",
        f"largest_map_coverage {largest_map_coverage:.9f}",
        f"largest_map_train_pose_frames {train_largest_pose_frames}",
        f"largest_map_test_pose_frames {test_largest_pose_frames}",
        f"train_metric_frames {train_metric_frames}",
        f"train_metric_coverage_selected {train_cov_selected:.9f}",
        f"train_metric_coverage_largest_map {train_cov_largest:.9f}",
        f"train_psnr {fmt(train_psnr, 9)}",
        f"train_ssim {fmt(train_ssim, 9)}",
        f"test_metric_frames {test_metric_frames}",
        f"test_metric_coverage_selected {test_cov_selected:.9f}",
        f"test_metric_coverage_largest_map {test_cov_largest:.9f}",
        f"test_psnr {fmt(test_psnr, 9)}",
        f"test_ssim {fmt(test_ssim, 9)}",
        f"ate_rmse_m {fmt(ate_rmse, 9)}",
        f"pipeline_time_sec {fmt(pipeline_sec, 9)}",
        f"pipeline_fps {fmt(fps, 9)}",
        f"final_gaussian_count {gaussian_count}",
        f"raw_train_metric_frames {len(raw_train_rows)}",
        f"raw_train_psnr {fmt(finite_mean(raw_train_rows, 'psnr'), 9)}",
        f"raw_train_ssim {fmt(finite_mean(raw_train_rows, 'ssim'), 9)}",
        f"raw_test_metric_frames {len(raw_test_rows)}",
        f"raw_test_psnr {fmt(finite_mean(raw_test_rows, 'psnr'), 9)}",
        f"raw_test_ssim {fmt(finite_mean(raw_test_rows, 'ssim'), 9)}",
    ]
    (result_dir / "split_benchmark_summary.txt").write_text("\n".join(summary_lines) + "\n")

    # One-row CSV for later concatenation across sequences/methods.
    csv_path = result_dir / "split_benchmark_summary.csv"
    fields = [
        "sequence", "input_frames", "train_selected", "test_selected",
        "largest_map_frames", "largest_map_coverage",
        "train_metric_frames", "train_psnr", "train_ssim", "train_coverage_selected",
        "test_metric_frames", "test_psnr", "test_ssim", "test_coverage_selected",
        "ate_rmse_m", "pipeline_time_sec", "fps", "gaussian_count",
    ]
    values = {
        "sequence": sequence,
        "input_frames": input_frames,
        "train_selected": len(train_ids),
        "test_selected": len(test_ids),
        "largest_map_frames": largest_pose_frames,
        "largest_map_coverage": fmt(largest_map_coverage, 9),
        "train_metric_frames": train_metric_frames,
        "train_psnr": fmt(train_psnr, 9),
        "train_ssim": fmt(train_ssim, 9),
        "train_coverage_selected": fmt(train_cov_selected, 9),
        "test_metric_frames": test_metric_frames,
        "test_psnr": fmt(test_psnr, 9),
        "test_ssim": fmt(test_ssim, 9),
        "test_coverage_selected": fmt(test_cov_selected, 9),
        "ate_rmse_m": fmt(ate_rmse, 9),
        "pipeline_time_sec": fmt(pipeline_sec, 9),
        "fps": fmt(fps, 9),
        "gaussian_count": gaussian_count,
    }
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(values)

    print("[Split benchmark summary]")
    print(f"  sequence        : {sequence}")
    print(f"  largest map     : {largest_pose_frames}/{input_frames} ({100*largest_map_coverage:.2f}%)")
    print(f"  train PSNR/SSIM : {fmt(train_psnr)} / {fmt(train_ssim)}  "
          f"[{train_metric_frames}/{len(train_ids)} selected]")
    print(f"  test  PSNR/SSIM : {fmt(test_psnr)} / {fmt(test_ssim)}  "
          f"[{test_metric_frames}/{len(test_ids)} selected]")
    print(f"  ATE RMSE        : {fmt(ate_rmse)} m")
    print(f"  pipeline FPS    : {fmt(fps, 3)}")
    print(f"  Gaussians       : {gaussian_count:,}")
    print(f"  output          : {result_dir / 'split_benchmark_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
