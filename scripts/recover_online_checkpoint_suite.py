#!/usr/bin/env python3
"""Recover ONLINE metrics for existing SH000-SH003 full-sequence runs.

No SLAM/mapping/optimization is rerun. This script only:
  1. renders each already-saved *_online checkpoint on recovered largest-map poses;
  2. reruns the lightweight summary step;
  3. aggregates ONLINE + FINAL_TAIL rows into one CSV.

Existing online_tracked_view_eval/metrics.csv is reused unless --force is given.
Failures are isolated per sequence so one problematic sequence does not discard
successfully recovered results from the others.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        parts = raw.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def upsert_kv(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(key + " "):
            out.append(f"{key} {value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key} {value}")
    path.write_text("\n".join(out) + "\n")


def ensure_online_timing_for_summary(result_dir: Path) -> None:
    timing_path = result_dir / "timing_summary.txt"
    timing = read_kv(timing_path)
    if "online_pipeline_wall_sec" in timing:
        return
    approx_sec = timing.get("online_pipeline_wall_sec_approx")
    approx_fps = timing.get("online_pipeline_fps_approx")
    if approx_sec is None:
        stream_sec = timing.get("stream_wall_sec")
        input_frames = timing.get("input_frames")
        if stream_sec is None or input_frames is None:
            return
        sec = float(stream_sec)
        n = int(input_frames)
        approx_sec = f"{sec:.9f}"
        approx_fps = f"{(n / sec if sec > 0 else float('nan')):.9f}"
    upsert_kv(timing_path, "online_pipeline_wall_sec", approx_sec)
    if approx_fps is not None:
        upsert_kv(timing_path, "online_pipeline_fps", approx_fps)
    upsert_kv(timing_path, "online_timing_is_approx", "1")
    upsert_kv(timing_path, "online_timing_source", "stream_wall_sec")
    upsert_kv(
        timing_path,
        "note_online_approx_excludes_tail_but_may_omit_mapper_catchup_after_last_track_call",
        "1",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="+", default=["SH000", "SH001", "SH002", "SH003"])
    ap.add_argument("--dataset-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo")
    ap.add_argument("--gt-root", default="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt")
    ap.add_argument("--output-root", default="./results")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    gt_root = Path(args.gt_root)

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for seq in args.sequences:
        result_dir = output_root / f"tartanair_v1_{seq}_0_full_split80_20_online_final"
        if not result_dir.exists():
            failures.append((seq, f"missing result directory: {result_dir}"))
            continue

        try:
            largest_summary = result_dir / "largest_map_eval" / "summary.txt"
            if not largest_summary.exists():
                run([
                    sys.executable, "scripts/evaluate_photoslam_largest_map.py",
                    "--result_dir", str(result_dir),
                    "--gt", str(gt_root / f"{seq}.txt"),
                    "--fps", str(args.fps),
                ], root)

            online_metrics = result_dir / "online_tracked_view_eval" / "metrics.csv"
            if args.force or not online_metrics.exists():
                run([
                    sys.executable, "scripts/recover_online_checkpoint_eval.py",
                    "--result-dir", str(result_dir),
                    "--sequence", str(dataset_root / seq),
                    "--fps", str(args.fps),
                ], root)
            else:
                print(f"\n[Skip recovery] {seq}: {online_metrics} already exists")

            ensure_online_timing_for_summary(result_dir)

            run([
                sys.executable, "scripts/summarize_tartanair_split_run.py",
                "--result-dir", str(result_dir),
                "--sequence", seq,
            ], root)

            with (result_dir / "split_benchmark_summary.csv").open(newline="") as f:
                rows.extend(csv.DictReader(f))
        except subprocess.CalledProcessError as e:
            failures.append((seq, f"command failed rc={e.returncode}"))
        except Exception as e:
            failures.append((seq, str(e)))

    aggregate = output_root / "tartanair_v1_SH000_SH003_0_full_split80_20_online_final_recovered_summary.csv"
    if rows:
        with aggregate.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n================ Recovered ONLINE / FINAL_TAIL benchmark ================")
    print("Sequence | Mode       | MaxMap | Train PSNR/SSIM | Test PSNR/SSIM | ATE(m) | FPS* | Gaussians")
    for r in rows:
        fps = "—" if r["mode"] != "ONLINE" or not r.get("fps") else f"{float(r['fps']):.2f}"
        print(
            f"{r['sequence']:7s} | {r['mode']:10s} | "
            f"{100*float(r['largest_map_coverage']):6.2f}% | "
            f"{float(r['train_psnr']):6.2f}/{float(r['train_ssim']):.4f} | "
            f"{float(r['test_psnr']):6.2f}/{float(r['test_ssim']):.4f} | "
            f"{float(r['ate_rmse_m']):.4f} | {fps:>5s} | {int(r['gaussian_count']):,}"
        )
    print("* Affected historical ONLINE runs use stream_wall_sec as an explicitly marked FPS approximation.")
    if rows:
        print(f"Aggregate CSV: {aggregate}")

    if failures:
        print("\n[Recovery failures]")
        for seq, reason in failures:
            print(f"  {seq}: {reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
