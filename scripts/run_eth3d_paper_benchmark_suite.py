#!/usr/bin/env python3
"""Final-only Photo-SLAM benchmark on rectified ETH3D stereo sequences.

Default sequences:
  mannequin_face_1, einstein_1, sofa_3, plant_scene_3

Input convention (produced by rectify_eth3d_stereo.py):
  image_left/             = rectified ETH3D camera 2 (physical left)
  image_right/            = rectified ETH3D camera 1 (physical right)
  calibration.json        = rectified K/P/baseline
  groundtruth_left.txt    = c2w of the rectified-left camera
  timestamps.txt          = original ETH3D image timestamps in seconds

The suite builds a temporary EuRoC-like symlink adapter because the already tested
Photo-SLAM held-out stereo runner consumes integer-nanosecond filenames. Actual
ETH3D timestamps are preserved exactly (to nanosecond rounding), so tracking and
trajectory/GT evaluation use the original timing.

One final Photo-SLAM result is reported. No ONLINE/FINAL_TAIL metric split is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

SEQUENCES = ["mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3"]


def run(cmd: list[str], cwd: Path, env=None) -> int:
    print("\n>>> " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def write_orb_yaml(calib: dict, timestamps: list[float], out: Path) -> None:
    K = calib["K_rectified_left"]
    K2 = calib["K_rectified_right"]
    if any(abs(float(K[r][c]) - float(K2[r][c])) > 1e-5 for r in range(3) for c in range(3)):
        raise RuntimeError("Rectified left/right K differ; expected CALIB_ZERO_DISPARITY output")
    width, height = map(int, calib["rectified_size"])
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    baseline = float(calib["baseline_rectified_m"])
    if baseline <= 0:
        raise RuntimeError(f"Invalid rectified baseline: {baseline}")
    if len(timestamps) >= 2:
        diffs = [b-a for a,b in zip(timestamps[:-1], timestamps[1:]) if b > a]
        fps = int(round(1.0 / (sorted(diffs)[len(diffs)//2]))) if diffs else 27
    else:
        fps = 27

    # Keep Photo-SLAM's official EuRoC ORB extractor values. Only the rectified
    # ETH3D camera geometry, image size, fps and stereo baseline are dataset-specific.
    # The runner loads PNGs with cv::imread(), hence input memory order is BGR.
    # Camera.RGB=0 makes Photo-SLAM convert BGR->RGB for its Gaussian color path.
    text = f'''%YAML:1.0
File.version: "1.0"

Camera.type: "Rectified"
Camera1.fx: {fx:.12f}
Camera1.fy: {fy:.12f}
Camera1.cx: {cx:.12f}
Camera1.cy: {cy:.12f}
Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps}
Camera.RGB: 0

Stereo.ThDepth: 60.0
Stereo.b: {baseline:.12f}

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
'''
    out.write_text(text)


def prepare_adapter(rectified: Path, result_dir: Path) -> tuple[Path, Path, list[float]]:
    left_dir = rectified / "image_left"
    right_dir = rectified / "image_right"
    calib = rectified / "calibration.json"
    gt = rectified / "groundtruth_left.txt"
    ts_file = rectified / "timestamps.txt"
    for p in (left_dir, right_dir, calib, gt, ts_file):
        if not p.exists():
            raise FileNotFoundError(f"Required ETH3D rectified input missing: {p}")

    raw_stems = [x.strip() for x in ts_file.read_text().splitlines() if x.strip()]
    timestamps = [float(x) for x in raw_stems]
    if any(b <= a for a,b in zip(timestamps[:-1], timestamps[1:])):
        raise RuntimeError(f"timestamps.txt is not strictly increasing: {ts_file}")

    adapter = result_dir / "_input_adapter"
    cam0 = adapter / "mav0" / "cam0" / "data"   # runner left
    cam1 = adapter / "mav0" / "cam1" / "data"   # runner right
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    ns_file = adapter / "timestamps_ns.txt"
    manifest = adapter / "frame_manifest.csv"

    used_ns: set[int] = set()
    with ns_file.open("w") as fts, manifest.open("w", newline="") as fm:
        w = csv.writer(fm)
        w.writerow(["frame_index", "timestamp", "timestamp_ns", "source_left", "source_right"])
        for i, (stem, ts) in enumerate(zip(raw_stems, timestamps)):
            src_l = (left_dir / f"{stem}.png").resolve()
            src_r = (right_dir / f"{stem}.png").resolve()
            if not src_l.exists() or not src_r.exists():
                raise FileNotFoundError(f"Missing rectified stereo pair: {stem}")
            ns = int(round(ts * 1e9))
            if ns in used_ns:
                raise RuntimeError(f"Timestamp collision after nanosecond conversion: {stem} -> {ns}")
            used_ns.add(ns)
            dst_l = cam0 / f"{ns}.png"
            dst_r = cam1 / f"{ns}.png"
            if dst_l.exists() or dst_l.is_symlink(): dst_l.unlink()
            if dst_r.exists() or dst_r.is_symlink(): dst_r.unlink()
            dst_l.symlink_to(os.path.relpath(src_l, start=cam0))
            dst_r.symlink_to(os.path.relpath(src_r, start=cam1))
            fts.write(f"{ns}\n")
            w.writerow([i, f"{ts:.9f}", ns, str(src_l), str(src_r)])
    return adapter, ns_file, timestamps


def preflight(root: Path) -> None:
    mapper_src = (root / "src" / "gaussian_mapper.cpp").read_text(errors="ignore")
    if "PAPER MINIMAL ONLINE checkpoint (evaluation only)" in mapper_src:
        raise RuntimeError(
            "Old ONLINE/FINAL_TAIL instrumentation is still applied to gaussian_mapper.cpp.\n"
            "For ETH3D final-only evaluation run:\n"
            "  git restore src/gaussian_mapper.cpp include/gaussian_mapper.h\n"
            "  python3 scripts/apply_gaussian_mapper_shutdown_guard.py\n"
            "  cmake --build build --target euroc_stereo_eval -j8"
        )
    tracking_h = (root / "ORB-SLAM3" / "include" / "Tracking.h").read_text(errors="ignore")
    if "SuppressKeyFrameInsertion" not in tracking_h:
        raise RuntimeError(
            "Held-out tracking-only support is missing. Run:\n"
            "  python3 scripts/apply_tartanair_heldout_split_patch.py\n"
            "  cmake --build ORB-SLAM3/build -j8\n"
            "  cmake --build build --target euroc_stereo_eval -j8"
        )
    binary = root / "bin" / "euroc_stereo_eval"
    if not binary.exists():
        raise FileNotFoundError(f"Missing {binary}; build target euroc_stereo_eval first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/shiyo/Desktop/Datasets/ETH3D_rectified")
    ap.add_argument("--sequences", nargs="+", default=SEQUENCES)
    ap.add_argument("--output-root", default="./results")
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--test-offset", type=int, default=4)
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    preflight(root)
    dataset_root = Path(args.dataset_root).resolve()
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dep = subprocess.run([sys.executable, "-c", "import torch, lpips, PIL; print('LPIPS dependency OK')"], cwd=str(root))
    if dep.returncode != 0:
        print("Install matching Python PyTorch + LPIPS before running the benchmark.")
        return dep.returncode

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    aggregate: list[dict] = []
    failures: list[tuple[str,str]] = []

    for seq in args.sequences:
        rectified = dataset_root / seq
        result_dir = output_root / f"eth3d_{seq}_split80_20_photoslam_final"
        if result_dir.exists():
            if not args.overwrite:
                failures.append((seq, f"result exists: {result_dir}; use --overwrite"))
                continue
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True)

        try:
            adapter, timestamps_ns, timestamps = prepare_adapter(rectified, result_dir)
            calib = json.loads((rectified / "calibration.json").read_text())
            orb_yaml = result_dir / "eth3d_rectified_orb.yaml"
            write_orb_yaml(calib, timestamps, orb_yaml)
        except Exception as e:
            failures.append((seq, f"input preparation failed: {e}"))
            continue

        cmd = [
            str(root / "bin" / "euroc_stereo_eval"),
            str(root / "ORB-SLAM3" / "Vocabulary" / "ORBvoc.txt"),
            str(orb_yaml),
            str(root / "cfg" / "gaussian_mapper" / "Stereo" / "EuRoC" / "EuRoC.yaml"),
            str(adapter),
            str(timestamps_ns),
            str(result_dir),
            "--stride=1",
            f"--test-every={args.test_every}",
            f"--test-offset={args.test_offset}",
        ]
        rc = run(cmd, root, env)
        if rc != 0:
            failures.append((seq, f"Photo-SLAM failed rc={rc}"))
            continue

        # The final-only protocol must not have generated the old pre-tail checkpoint.
        if (result_dir / "online_checkpoint_metadata.txt").exists():
            failures.append((seq, "unexpected ONLINE checkpoint was generated; local mapper is not final-only"))
            continue

        rc = run([
            sys.executable, "scripts/add_lpips_to_photoslam_metrics.py",
            "--result-dir", str(result_dir),
            "--net", args.lpips_net,
            "--subdirs", "final_tracked_view_eval",
        ], root, env)
        if rc != 0:
            failures.append((seq, f"LPIPS failed rc={rc}"))
            continue

        rc = run([
            sys.executable, "scripts/summarize_eth3d_paper_run.py",
            "--result-dir", str(result_dir),
            "--sequence", seq,
            "--groundtruth", str(rectified / "groundtruth_left.txt"),
        ], root)
        if rc != 0:
            failures.append((seq, f"summary/ATE failed rc={rc}"))
            continue

        with (result_dir / "benchmark_summary.csv").open(newline="") as f:
            aggregate.extend(csv.DictReader(f))

    if aggregate:
        aggregate_path = output_root / ("eth3d_" + "_".join(args.sequences) + "_photoslam_final_summary.csv")
        with aggregate_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
            w.writeheader(); w.writerows(aggregate)

        print("\n================ Photo-SLAM ETH3D FINAL ================")
        print("Sequence          | MaxMap | Train P/S/L              | Test P/S/L               | ATE(m) | FPS   | Total(s) | Gaussians")
        print("-" * 130)
        for r in aggregate:
            tr = f"{float(r['train_psnr']):.2f}/{float(r['train_ssim']):.4f}/{float(r['train_lpips']):.4f}"
            te = f"{float(r['test_psnr']):.2f}/{float(r['test_ssim']):.4f}/{float(r['test_lpips']):.4f}"
            print(f"{r['sequence']:17s} | {100*float(r['largest_map_coverage']):6.2f}% | {tr:24s} | {te:24s} | {float(r['ate_rmse_m']):6.4f} | {float(r['fps']):5.2f} | {float(r['total_time_sec']):8.2f} | {int(float(r['gaussian_count'])):,}")
        print(f"Aggregate CSV: {aggregate_path}")

    if failures:
        print("\n[Failures]")
        for seq, reason in failures:
            print(f"  {seq}: {reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
