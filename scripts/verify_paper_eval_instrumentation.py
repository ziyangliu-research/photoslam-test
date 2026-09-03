#!/usr/bin/env python3
"""Strict preflight for the minimal Photo-SLAM paper evaluation path."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"
BINARY = ROOT / "bin" / "tartanair_stereo_eval"


def main() -> int:
    errors: list[str] = []

    mapper = MAPPER.read_text(errors="ignore") if MAPPER.exists() else ""
    runner = RUNNER.read_text(errors="ignore") if RUNNER.exists() else ""

    required_mapper = [
        "PAPER MINIMAL ONLINE checkpoint (evaluation only)",
        "PAPER TAIL timing instrumentation (measurement only)",
        "offline_tail_metadata.txt",
    ]
    required_runner = [
        "Held-out NVS split",
        "Split all-view final evaluation",
        "PAPER MINIMAL ONLINE vs FINAL_TAIL evaluation",
        "online_pipeline_wall_sec",
        "online_tracked_view_eval",
        "final_tracked_view_eval",
    ]
    forbidden = [
        "online_frame_poses.csv",
        "loadOnlineFramePoseSnapshot",
        "ONLINE frame-pose snapshot (evaluation only)",
    ]

    for token in required_mapper:
        if token not in mapper:
            errors.append(f"mapper missing {token!r}")
    for token in required_runner:
        if token not in runner:
            errors.append(f"runner missing {token!r}")
    for token in forbidden:
        if token in mapper or token in runner:
            errors.append(f"non-minimal/v2 instrumentation still present: {token!r}")

    if not BINARY.exists():
        errors.append(f"missing binary: {BINARY}")
    else:
        proc = subprocess.run(["strings", str(BINARY)], capture_output=True, text=True)
        b = proc.stdout
        for token in [
            "online_pipeline_wall_sec",
            "online_tracked_view_eval",
            "offline_tail_metadata.txt",
        ]:
            if token not in b:
                errors.append(f"binary stale: missing {token!r}")
        for token in ["online_frame_poses.csv", "loadOnlineFramePoseSnapshot"]:
            if token in b:
                errors.append(f"binary contains forbidden v2 token {token!r}")

    if errors:
        print("[PAPER evaluation preflight] FAILED")
        for e in errors:
            print("  -", e)
        return 1

    print("[PAPER evaluation preflight] OK")
    print("  held-out split                     : present")
    print("  minimal ONLINE/FINAL_TAIL eval     : present")
    print("  original tail measurement timer    : present")
    print("  v2/recovery pose instrumentation   : absent")
    print("  built binary                       : up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
