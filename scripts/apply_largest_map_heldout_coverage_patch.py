#!/usr/bin/env python3
"""Make largest-map render coverage use held-out test views as denominator.

When test_frame_ids.txt exists, PSNR/SSIM coverage is reported against the
held-out test set rather than against all input frames. Pose/ATE coverage remains
against all selected input frames.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "evaluate_photoslam_largest_map.py"
MARKER = "render_metric_denominator_type"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Patch anchor not found for {label}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text()
    if MARKER in text:
        print(f"Already patched: {PATH}")
        return 0

    anchor = '''    selected_end = int(
        tracking_summary.get("selected_end_frame", largest_frame_ids[-1])
    )

'''
    block = '''    # Held-out NVS runs write exact test ids. Pose/ATE coverage still uses all
    # selected input frames, while render coverage should use only the test-set
    # denominator (e.g. 40 held-out views for a 200-frame 8:2 split).
    test_ids_path = result_dir / "test_frame_ids.txt"
    heldout_test_ids = []
    if test_ids_path.exists():
        heldout_test_ids = [
            int(line.strip())
            for line in test_ids_path.read_text().splitlines()
            if line.strip()
        ]
    render_metric_denominator = (
        len(heldout_test_ids) if heldout_test_ids else selected_input_frames
    )
    render_metric_denominator_type = (
        "heldout_test_frames" if heldout_test_ids else "selected_input_frames"
    )

'''
    text = replace_once(text, anchor, anchor + block, "held-out denominator setup")

    old = '        f"largest_map_render_metric_frames {len(filtered_rows)}",\n        f"largest_map_render_metric_coverage {len(filtered_rows) / selected_input_frames:.9f}",\n'
    new = '        f"largest_map_render_metric_frames {len(filtered_rows)}",\n        f"render_metric_denominator_type {render_metric_denominator_type}",\n        f"render_metric_denominator {render_metric_denominator}",\n        f"largest_map_render_metric_coverage {len(filtered_rows) / render_metric_denominator:.9f}",\n'
    text = replace_once(text, old, new, "summary render coverage")

    old = '    print(f"  render metric frames  : {len(filtered_rows)}")\n'
    new = '    print(f"  render metric frames  : {len(filtered_rows)}/{render_metric_denominator} "\n          f"({100.0 * len(filtered_rows) / render_metric_denominator:.2f}%)")\n'
    text = replace_once(text, old, new, "console render coverage")

    PATH.write_text(text)
    print(f"Patched: {PATH}")
    print("Held-out render coverage now uses test_frame_ids.txt when present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
