#!/usr/bin/env python3
"""Extend the held-out TartanAir runner to render both TRAIN and TEST views.

Mapping semantics are unchanged:
  * test frames remain pose-only and cannot create persistent ORB/Gaussian KFs;
  * train frames map normally.

This patch changes only the *offline final evaluation* after mapper timing has
already been recorded: evaluateFinalTrackedViews() receives all selected frames
instead of test frames only. The resulting metrics.csv can then be split into
train/test metrics by exact frame IDs without another Photo-SLAM run.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples" / "tartanair_stereo_eval.cpp"
MARKER = "Split all-view final evaluation"


def main() -> int:
    text = PATH.read_text()
    if MARKER in text:
        print(f"Already patched: {PATH}")
        return 0

    old = '''                evaluateFinalTrackedViews(
                    evaluation_frames,
                    tracking_records,
'''
    new = '''                // Split all-view final evaluation: render all final evaluable poses
                // after timing has stopped, then separate train/test metrics offline.
                evaluateFinalTrackedViews(
                    frames,
                    tracking_records,
'''
    if old not in text:
        # A clean/no-split runner may still pass frames already; in that case just
        # require the held-out split marker so we do not silently patch wrong code.
        if 'Held-out NVS split' in text and 'evaluateFinalTrackedViews(\n                    frames,' in text:
            print(f"All-view evaluation already active: {PATH}")
            return 0
        raise RuntimeError(
            "Could not find held-out evaluation call using evaluation_frames. "
            "Apply apply_tartanair_heldout_split_patch.py first."
        )

    text = text.replace(old, new, 1)
    PATH.write_text(text)
    print(f"Patched: {PATH}")
    print("Final offline rendering now covers all evaluable train+test views.")
    print("Internal stream/pipeline timing remains recorded before this rendering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
