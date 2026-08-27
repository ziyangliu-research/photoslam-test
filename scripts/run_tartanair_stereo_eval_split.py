#!/usr/bin/env python3
"""Run Photo-SLAM with a held-out TartanAir train/test split.

Default split:
  test_every=5, test_offset=4 -> 4,9,14,... are TEST frames.

TEST frames are still passed through the normal TrackStereo pose-estimation path.
The patched C++ runner only suppresses persistent ORB keyframe/map insertion for
those frames, so their images cannot enter Photo-SLAM Gaussian mapping/training.
Final PSNR/SSIM rendering is restricted to the held-out TEST frames.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from run_tartanair_stereo_eval import find_final_gaussian_ply, upsert_key_value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--orb-config", required=True)
    ap.add_argument("--binary", default="./bin/tartanair_stereo_eval")
    ap.add_argument("--vocab", default="./ORB-SLAM3/Vocabulary/ORBvoc.txt")
    ap.add_argument(
        "--gaussian-config",
        default="./cfg/gaussian_mapper/Stereo/TartanAir/TartanAir_stereo_eval.yaml",
    )
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--test-offset", type=int, default=4)

    rg = ap.add_mutually_exclusive_group()
    rg.add_argument("--end", type=int, default=None)
    rg.add_argument("--num-frames", type=int, default=None)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--skip-final-eval", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.end is not None and args.end < args.start:
        raise ValueError("--end must be >= --start")
    if args.num_frames is not None and args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if args.test_every <= 0:
        raise ValueError("--test-every must be > 0 in split mode")
    if not 0 <= args.test_offset < args.test_every:
        raise ValueError("--test-offset must satisfy 0 <= offset < test-every")

    result_dir = Path(args.output).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.binary,
        args.vocab,
        args.orb_config,
        args.gaussian_config,
        args.sequence,
        args.output,
        f"--fps={args.fps}",
        f"--start={args.start}",
        f"--test-every={args.test_every}",
        f"--test-offset={args.test_offset}",
    ]
    if args.end is not None:
        cmd.append(f"--end={args.end}")
    elif args.num_frames is not None:
        cmd.append(f"--num-frames={args.num_frames}")
    if args.viewer:
        cmd.append("viewer")
    if args.skip_final_eval:
        cmd.append("--skip-final-eval")

    print("[Photo-SLAM held-out wrapper] Running:")
    print("  " + " ".join(cmd), flush=True)
    print(
        f"[Split] every={args.test_every}, offset={args.test_offset}; "
        "test frames keep normal pose tracking but cannot create mapping keyframes",
        flush=True,
    )

    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        return completed.returncode

    try:
        ply_path, gaussian_count = find_final_gaussian_ply(result_dir)
        initialized = 1
        source_ply = str(ply_path)
    except FileNotFoundError:
        gaussian_count = 0
        initialized = 0
        source_ply = "none"

    (result_dir / "final_gaussian_count.txt").write_text(
        f"gaussian_map_initialized {initialized}\n"
        f"final_gaussian_count {gaussian_count}\n"
        f"source_ply {source_ply}\n"
    )
    upsert_key_value(result_dir / "tracking_summary.txt", "gaussian_map_initialized", str(initialized))
    upsert_key_value(result_dir / "tracking_summary.txt", "final_gaussian_count", str(gaussian_count))
    upsert_key_value(result_dir / "timing_summary.txt", "final_gaussian_count", str(gaussian_count))
    upsert_key_value(
        result_dir / "timing_summary.txt",
        "note_gaussian_count_postprocess_excluded_from_internal_timing",
        "1",
    )

    test_ids_path = result_dir / "test_frame_ids.txt"
    train_ids_path = result_dir / "train_frame_ids.txt"
    test_ids = [x for x in test_ids_path.read_text().splitlines() if x.strip()] if test_ids_path.exists() else []
    train_ids = [x for x in train_ids_path.read_text().splitlines() if x.strip()] if train_ids_path.exists() else []

    print("[Held-out split result]")
    print(f"  train frames    : {len(train_ids)}")
    print(f"  test frames     : {len(test_ids)}")
    print(f"  Gaussian count  : {gaussian_count:,}")
    print(f"  train ids       : {train_ids_path}")
    print(f"  test ids        : {test_ids_path}")
    if not args.skip_final_eval:
        print(f"  test-view eval  : {result_dir / 'final_tracked_view_eval' / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
