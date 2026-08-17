#!/usr/bin/env python3

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    root = Path(args.sequence_root).resolve()
    left_dir = root / "image_lcam_front"
    right_dir = root / "image_rcam_front"

    out = Path(args.output_dir).resolve()
    cam0 = out / "mav0" / "cam0" / "data"
    cam1 = out / "mav0" / "cam1" / "data"

    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob("*_lcam_front.png"))
    if not left_files:
        raise RuntimeError(f"No left images found in {left_dir}")

    indices = []
    for p in left_files:
        idx = int(p.name.split("_")[0])
        if idx < args.start:
            continue
        if args.end >= 0 and idx > args.end:
            continue
        indices.append(idx)

    if not indices:
        raise RuntimeError("No frames selected.")

    timestamps = []
    mapping = []

    for idx in indices:
        left = left_dir / f"{idx:06d}_lcam_front.png"
        right = right_dir / f"{idx:06d}_rcam_front.png"

        if not left.exists():
            raise FileNotFoundError(left)
        if not right.exists():
            raise FileNotFoundError(right)

        # TartanAir v2 raw RGB is sampled at 10 Hz.
        timestamp_ns = round(idx / args.fps * 1e9)
        stamp = f"{timestamp_ns:019d}"

        dst_left = cam0 / f"{stamp}.png"
        dst_right = cam1 / f"{stamp}.png"

        if dst_left.exists() or dst_left.is_symlink():
            dst_left.unlink()
        if dst_right.exists() or dst_right.is_symlink():
            dst_right.unlink()

        dst_left.symlink_to(left)
        dst_right.symlink_to(right)

        timestamps.append(stamp)
        mapping.append(
            f"{idx:06d} {stamp} {left.name} {right.name}"
        )

    (out / "timestamps.txt").write_text(
        "\n".join(timestamps) + "\n"
    )

    (out / "frame_mapping.txt").write_text(
        "\n".join(mapping) + "\n"
    )

    print(f"Prepared {len(indices)} stereo frames")
    print(f"Range: {indices[0]} -> {indices[-1]}")
    print(f"Adapter: {out}")
    print(f"Timestamps: {out / 'timestamps.txt'}")


if __name__ == "__main__":
    main()
