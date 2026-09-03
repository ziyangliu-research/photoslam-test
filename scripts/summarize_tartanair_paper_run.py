#!/usr/bin/env python3
"""Paper summary for Photo-SLAM ONLINE vs FINAL_TAIL held-out evaluation.

Required inputs under result_dir:
  online_tracked_view_eval/metrics.csv   (PSNR, SSIM, LPIPS)
  final_tracked_view_eval/metrics.csv    (PSNR, SSIM, LPIPS)
  train_frame_ids.txt / test_frame_ids.txt
  largest_map_eval/frame_ids.txt + summary.txt
  timing_summary.txt                     (exact ONLINE time)
  offline_tail_metadata.txt              (exact original tail-loop time)
  *_online and *_shutdown PLYs

Timing policy:
  ONLINE      : FPS reported, offline=0, total=online.
  FINAL_TAIL  : FPS not reported, online is shared, offline is the original
                Photo-SLAM tail-loop time, total=online+offline.

Evaluation/checkpoint rendering and PLY I/O are excluded from these paper times.
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
        if text in (None, ""):
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
    candidates = list(result_dir.glob(f"*_{suffix}/ply/point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        return None

    def iteration_of(p: Path) -> int:
        try:
            return int(p.parent.name.split("_", 1)[1])
        except Exception:
            return -1

    return max(candidates, key=iteration_of)


def load_metrics(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing view metrics: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "lpips" not in rows[0]:
        raise RuntimeError(
            f"LPIPS column missing from {path}. Run scripts/add_lpips_to_photoslam_metrics.py first."
        )
    return rows


def summarize_mode(
    rows: List[dict],
    train_set: Set[int],
    test_set: Set[int],
    largest_set: Set[int],
    train_selected: int,
    test_selected: int,
) -> dict:
    largest_train_ids = train_set & largest_set
    largest_test_ids = test_set & largest_set
    train_rows = [r for r in rows if int(r["frame_index"]) in largest_train_ids]
    test_rows = [r for r in rows if int(r["frame_index"]) in largest_test_ids]
    return {
        "train_metric_frames": len(train_rows),
        "train_psnr": finite_mean(train_rows, "psnr"),
        "train_ssim": finite_mean(train_rows, "ssim"),
        "train_lpips": finite_mean(train_rows, "lpips"),
        "train_coverage_selected": len(train_rows) / train_selected if train_selected else 0.0,
        "test_metric_frames": len(test_rows),
        "test_psnr": finite_mean(test_rows, "psnr"),
        "test_ssim": finite_mean(test_rows, "ssim"),
        "test_lpips": finite_mean(test_rows, "lpips"),
        "test_coverage_selected": len(test_rows) / test_selected if test_selected else 0.0,
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
        raise FileNotFoundError("largest_map_eval/frame_ids.txt missing or empty")

    train_set, test_set, largest_set = set(train_ids), set(test_ids), set(largest_ids)
    online_rows = load_metrics(result_dir / "online_tracked_view_eval" / "metrics.csv")
    final_rows = load_metrics(result_dir / "final_tracked_view_eval" / "metrics.csv")
    online_metrics = summarize_mode(online_rows, train_set, test_set, largest_set, len(train_ids), len(test_ids))
    final_metrics = summarize_mode(final_rows, train_set, test_set, largest_set, len(train_ids), len(test_ids))

    tracking = read_kv(result_dir / "tracking_summary.txt")
    timing = read_kv(result_dir / "timing_summary.txt")
    tail_timing = read_kv(result_dir / "offline_tail_metadata.txt")
    largest_summary = read_kv(result_dir / "largest_map_eval" / "summary.txt")

    input_frames = int(tracking.get("input_frames", len(train_ids) + len(test_ids)))
    largest_frames = int(largest_summary.get("largest_map_pose_frames", len(largest_ids)))
    largest_cov = float(largest_summary.get("largest_map_pose_coverage", largest_frames / input_frames))
    ate = float(largest_summary.get("ate_rmse_m", "nan"))

    if "online_pipeline_wall_sec" not in timing:
        raise RuntimeError("Exact online_pipeline_wall_sec missing from timing_summary.txt")
    if "offline_tail_optimization_wall_sec" not in tail_timing:
        raise RuntimeError("offline_tail_optimization_wall_sec missing from offline_tail_metadata.txt")

    online_sec = float(timing["online_pipeline_wall_sec"])
    tail_sec = float(tail_timing["offline_tail_optimization_wall_sec"])
    online_fps = input_frames / online_sec if online_sec > 0 else float("nan")
    final_total_sec = online_sec + tail_sec

    online_ply = find_checkpoint_ply(result_dir, "online")
    final_ply = find_checkpoint_ply(result_dir, "shutdown")
    if online_ply is None or final_ply is None:
        raise FileNotFoundError("ONLINE or FINAL_TAIL checkpoint PLY missing")
    online_g = read_ply_vertex_count(online_ply)
    final_g = read_ply_vertex_count(final_ply)

    modes = [
        {
            "mode": "ONLINE", "metrics": online_metrics, "fps": online_fps,
            "online_sec": online_sec, "offline_sec": 0.0, "total_sec": online_sec,
            "gaussians": online_g,
        },
        {
            "mode": "FINAL_TAIL", "metrics": final_metrics, "fps": float("nan"),
            "online_sec": online_sec, "offline_sec": tail_sec, "total_sec": final_total_sec,
            "gaussians": final_g,
        },
    ]

    summary_lines = [
        f"sequence {sequence}",
        f"input_frames {input_frames}",
        f"train_selected_frames {len(train_ids)}",
        f"test_selected_frames {len(test_ids)}",
        f"largest_map_pose_frames {largest_frames}",
        f"largest_map_coverage {largest_cov:.9f}",
        f"ate_rmse_m {fmt(ate, 9)}",
        "fps_policy ONLINE_only",
        "time_policy compute_only_excluding_checkpoint_io_shutdown_save_and_metric_rendering",
        f"shared_online_time_sec {fmt(online_sec, 9)}",
        f"offline_tail_optimization_time_sec {fmt(tail_sec, 9)}",
        "",
    ]
    for item in modes:
        mode, m = item["mode"], item["metrics"]
        prefix = mode.lower()
        summary_lines += [
            f"[{mode}]",
            f"{prefix}_train_metric_frames {m['train_metric_frames']}",
            f"{prefix}_train_psnr {fmt(m['train_psnr'], 9)}",
            f"{prefix}_train_ssim {fmt(m['train_ssim'], 9)}",
            f"{prefix}_train_lpips {fmt(m['train_lpips'], 9)}",
            f"{prefix}_test_metric_frames {m['test_metric_frames']}",
            f"{prefix}_test_psnr {fmt(m['test_psnr'], 9)}",
            f"{prefix}_test_ssim {fmt(m['test_ssim'], 9)}",
            f"{prefix}_test_lpips {fmt(m['test_lpips'], 9)}",
            f"{prefix}_fps {fmt(item['fps'], 9)}",
            f"{prefix}_online_time_sec {fmt(item['online_sec'], 9)}",
            f"{prefix}_offline_opt_time_sec {fmt(item['offline_sec'], 9)}",
            f"{prefix}_total_time_sec {fmt(item['total_sec'], 9)}",
            f"{prefix}_gaussian_count {item['gaussians']}",
            "",
        ]
    (result_dir / "split_benchmark_summary.txt").write_text("\n".join(summary_lines) + "\n")

    fields = [
        "sequence", "mode", "input_frames", "train_selected", "test_selected",
        "largest_map_frames", "largest_map_coverage",
        "train_metric_frames", "train_psnr", "train_ssim", "train_lpips",
        "test_metric_frames", "test_psnr", "test_ssim", "test_lpips",
        "ate_rmse_m", "fps", "online_time_sec", "offline_opt_time_sec",
        "total_time_sec", "gaussian_count",
    ]
    csv_path = result_dir / "split_benchmark_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in modes:
            m = item["metrics"]
            writer.writerow({
                "sequence": sequence,
                "mode": item["mode"],
                "input_frames": input_frames,
                "train_selected": len(train_ids),
                "test_selected": len(test_ids),
                "largest_map_frames": largest_frames,
                "largest_map_coverage": fmt(largest_cov, 9),
                "train_metric_frames": m["train_metric_frames"],
                "train_psnr": fmt(m["train_psnr"], 9),
                "train_ssim": fmt(m["train_ssim"], 9),
                "train_lpips": fmt(m["train_lpips"], 9),
                "test_metric_frames": m["test_metric_frames"],
                "test_psnr": fmt(m["test_psnr"], 9),
                "test_ssim": fmt(m["test_ssim"], 9),
                "test_lpips": fmt(m["test_lpips"], 9),
                "ate_rmse_m": fmt(ate, 9),
                "fps": fmt(item["fps"], 9) if item["mode"] == "ONLINE" else "",
                "online_time_sec": fmt(item["online_sec"], 9),
                "offline_opt_time_sec": fmt(item["offline_sec"], 9),
                "total_time_sec": fmt(item["total_sec"], 9),
                "gaussian_count": item["gaussians"],
            })

    print("[Photo-SLAM PAPER ONLINE vs FINAL_TAIL]")
    print(f"  sequence    : {sequence}")
    print(f"  largest map : {largest_frames}/{input_frames} ({100*largest_cov:.2f}%)")
    print(f"  ATE RMSE    : {fmt(ate)} m")
    for item in modes:
        m = item["metrics"]
        print(f"  {item['mode']}")
        print(f"    Train P/S/L : {fmt(m['train_psnr'])} / {fmt(m['train_ssim'])} / {fmt(m['train_lpips'])}")
        print(f"    Test  P/S/L : {fmt(m['test_psnr'])} / {fmt(m['test_ssim'])} / {fmt(m['test_lpips'])}")
        print(f"    FPS         : {fmt(item['fps'], 3) if item['mode'] == 'ONLINE' else '—'}")
        print(f"    Time O/Off/T: {fmt(item['online_sec'], 3)} / {fmt(item['offline_sec'], 3)} / {fmt(item['total_sec'], 3)} s")
        print(f"    Gaussians   : {item['gaussians']:,}")
    print(f"  output      : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
