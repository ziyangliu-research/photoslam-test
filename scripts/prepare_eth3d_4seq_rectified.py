#!/usr/bin/env python3
"""Batch-rectify the four ETH3D stereo sequences used in the paper benchmark."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_SEQUENCES = [
    "mannequin_face_1",
    "einstein_1",
    "sofa_3",
    "plant_scene_3",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        default="/home/shiyo/Desktop/Datasets/ETH3D/sequences",
    )
    ap.add_argument(
        "--output-root",
        default="/home/shiyo/Desktop/Datasets/ETH3D_rectified",
    )
    ap.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing rectified sequence directory.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rectifier = repo_root / "scripts" / "rectify_eth3d_stereo.py"
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for seq in args.sequences:
        src = raw_root / seq / seq
        dst = output_root / seq
        if not src.is_dir():
            raise FileNotFoundError(f"ETH3D raw sequence not found: {src}")

        cmd = [
            sys.executable,
            str(rectifier),
            "--input",
            str(src),
            "--output",
            str(dst),
            "--alpha",
            str(args.alpha),
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        print("\n============================================================", flush=True)
        print(f"Rectifying {seq}", flush=True)
        print(" ".join(cmd), flush=True)
        print("============================================================", flush=True)
        rc = subprocess.run(cmd, cwd=str(repo_root)).returncode
        if rc != 0:
            print(f"FAILED: {seq} (exit={rc})", file=sys.stderr)
            return rc

    print("\nAll requested ETH3D sequences were rectified successfully.")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
