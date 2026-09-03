#!/usr/bin/env python3
"""Verify exact ONLINE/FINAL_TAIL instrumentation before expensive benchmark runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_text(path: Path, tokens: list[str]) -> list[str]:
    if not path.exists():
        return [f"missing file: {path}"]
    text = path.read_text(errors="ignore")
    return [f"{path}: missing token {t!r}" for t in tokens if t not in text]


def main() -> int:
    errors = []
    errors += require_text(ROOT / "src/gaussian_mapper.cpp", [
        "ONLINE checkpoint instrumentation (evaluation only)",
        "ONLINE frame-pose snapshot (evaluation only)",
        "online_checkpoint_metadata.txt",
        "online_frame_poses.csv",
        "_online",
    ])
    errors += require_text(ROOT / "examples/tartanair_stereo_eval.cpp", [
        "ONLINE vs FINAL_TAIL offline evaluation",
        "loadOnlineFramePoseSnapshot",
        "online_pipeline_wall_sec",
        "online_frame_poses.csv",
        "online_tracked_view_eval",
    ])

    binary = ROOT / "bin/tartanair_stereo_eval"
    if not binary.exists():
        errors.append(f"missing binary: {binary}")
    else:
        proc = subprocess.run(["strings", str(binary)], capture_output=True, text=True)
        binary_text = proc.stdout
        for token in [
            "online_pipeline_wall_sec",
            "online_frame_poses.csv",
            "online_tracked_view_eval",
        ]:
            if token not in binary_text:
                errors.append(f"binary is stale: missing {token!r}")

    if errors:
        print("[ONLINE/FINAL_TAIL preflight] FAILED")
        for e in errors:
            print("  -", e)
        print("Run scripts/repair_online_final_instrumentation_v2.py, rebuild tartanair_stereo_eval, then verify again.")
        return 1

    print("[ONLINE/FINAL_TAIL preflight] OK")
    print("  mapper ONLINE checkpoint          : present")
    print("  exact pre-tail frame-pose snapshot: present")
    print("  runner dual evaluation/timing     : present")
    print("  built binary                      : up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
