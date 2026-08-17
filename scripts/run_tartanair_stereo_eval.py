#!/usr/bin/env python3
"""Run tartanair_stereo_eval and report the final Gaussian count.

The wrapper keeps Photo-SLAM's existing runner behavior unchanged, then reads the
vertex count from the final *_shutdown PLY header. The count is therefore
available in both quality mode and --skip-final-eval timing mode.

The runner's internal timing_summary.txt remains the source of truth for timing:
this post-run Gaussian-count parsing is NOT included in those wall-time values.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True,
                    help="TartanAir V1/V2 sequence root")
    ap.add_argument("--output", required=True,
                    help="Photo-SLAM result directory")
    ap.add_argument("--orb-config", required=True,
                    help="ORB-SLAM3 camera settings YAML (V1/V2 specific)")
    ap.add_argument("--binary", default="./bin/tartanair_stereo_eval")
    ap.add_argument("--vocab", default="./ORB-SLAM3/Vocabulary/ORBvoc.txt")
    ap.add_argument(
        "--gaussian-config",
        default="./cfg/gaussian_mapper/Stereo/TartanAir/TartanAir_stereo_eval.yaml",
    )
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=0)

    range_group = ap.add_mutually_exclusive_group()
    range_group.add_argument("--end", type=int, default=None)
    range_group.add_argument("--num-frames", type=int, default=None)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="skip the added final tracked-view render/PSNR/SSIM evaluation",
    )
    return ap.parse_args()


def parse_ply_vertex_count(path: Path) -> Optional[int]:
    """Read only the ASCII PLY header; works for binary and ASCII PLY bodies."""
    with path.open("rb") as f:
        for _ in range(4096):
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="ignore").strip()
            m = re.fullmatch(r"element\s+vertex\s+(\d+)", line)
            if m:
                return int(m.group(1))
            if line == "end_header":
                break
    return None


def find_final_gaussian_ply(result_dir: Path) -> Tuple[Path, int]:
    shutdown_dirs = []
    for p in result_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"(\d+)_shutdown", p.name)
        if m:
            shutdown_dirs.append((int(m.group(1)), p))

    if not shutdown_dirs:
        raise FileNotFoundError(
            f"No <iteration>_shutdown directory found under {result_dir}"
        )

    shutdown_dirs.sort(key=lambda x: x[0], reverse=True)
    _, final_dir = shutdown_dirs[0]

    candidates = sorted(final_dir.rglob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No PLY found under final shutdown dir: {final_dir}")

    for ply in candidates:
        count = parse_ply_vertex_count(ply)
        if count is not None:
            return ply, count

    raise RuntimeError(
        f"Found PLY file(s) under {final_dir}, but no 'element vertex N' header"
    )


def upsert_key_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    prefix = key + " "
    replaced = False
    out = []
    for line in lines:
        if line.startswith(prefix):
            if not replaced:
                out.append(f"{key} {value}")
                replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key} {value}")
    path.write_text("\n".join(out) + "\n")


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
    ]
    if args.end is not None:
        cmd.append(f"--end={args.end}")
    elif args.num_frames is not None:
        cmd.append(f"--num-frames={args.num_frames}")
    if args.viewer:
        cmd.append("viewer")
    if args.skip_final_eval:
        cmd.append("--skip-final-eval")

    print("[Photo-SLAM wrapper] Running:")
    print("  " + " ".join(cmd), flush=True)

    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        return completed.returncode

    ply_path, gaussian_count = find_final_gaussian_ply(result_dir)

    (result_dir / "final_gaussian_count.txt").write_text(
        f"final_gaussian_count {gaussian_count}\n"
        f"source_ply {ply_path}\n"
    )

    # Keep the key statistic next to the existing run summaries as well.
    upsert_key_value(
        result_dir / "tracking_summary.txt",
        "final_gaussian_count",
        str(gaussian_count),
    )
    upsert_key_value(
        result_dir / "timing_summary.txt",
        "final_gaussian_count",
        str(gaussian_count),
    )
    upsert_key_value(
        result_dir / "timing_summary.txt",
        "note_gaussian_count_postprocess_excluded_from_internal_timing",
        "1",
    )

    print("[Final Gaussian model]")
    print(f"  Gaussian count : {gaussian_count:,}")
    print(f"  source PLY     : {ply_path}")
    print(f"  saved          : {result_dir / 'final_gaussian_count.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
