#!/usr/bin/env python3
"""Strict preflight for the minimal Photo-SLAM paper evaluation path.

Runner-only strings live in the tartanair_stereo_eval executable, while
GaussianMapper implementation strings normally live in the linked
libgaussian_mapper shared library. Check the correct linked artifact instead of
requiring mapper literals to be embedded in the executable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"
BINARY = ROOT / "bin" / "tartanair_stereo_eval"


def strings_of(path: Path) -> str:
    proc = subprocess.run(["strings", str(path)], capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else ""


def linked_local_artifacts(binary: Path) -> list[Path]:
    """Return Photo-SLAM-local shared objects resolved by ldd."""
    proc = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
    out: list[Path] = []
    root = ROOT.resolve()
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if "=>" not in line:
            continue
        rhs = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        if not rhs.startswith("/"):
            continue
        p = Path(rhs).resolve()
        if not p.exists():
            continue
        try:
            p.relative_to(root)
        except ValueError:
            continue
        out.append(p)
    return out


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
        binary_text = strings_of(BINARY)

        # These are emitted by tartanair_stereo_eval.cpp itself and therefore
        # must be present in the executable.
        for token in ["online_pipeline_wall_sec", "online_tracked_view_eval"]:
            if token not in binary_text:
                errors.append(f"runner binary stale: missing {token!r}")

        # GaussianMapper is linked as a shared library, so mapper-side literals
        # (including offline_tail_metadata.txt) normally live there rather than
        # in the final executable.
        local_libs = linked_local_artifacts(BINARY)
        linked_text = binary_text + "\n" + "\n".join(strings_of(p) for p in local_libs)
        if "offline_tail_metadata.txt" not in linked_text:
            libs = ", ".join(str(p) for p in local_libs) or "<no local linked libraries found>"
            errors.append(
                "linked Photo-SLAM artifacts stale: missing 'offline_tail_metadata.txt' "
                f"(checked: {libs})"
            )

        for token in ["online_frame_poses.csv", "loadOnlineFramePoseSnapshot"]:
            if token in linked_text:
                errors.append(f"built artifacts contain forbidden v2 token {token!r}")

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
    print("  built executable/shared libraries  : up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
