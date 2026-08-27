#!/usr/bin/env python3
"""Make largest-map image evaluation held-out-aware.

For held-out split runs (test_frame_ids.txt exists):
  * pose/ATE coverage remains against ALL selected input frames;
  * largest-map PSNR/SSIM rows are filtered to TEST frames only;
  * render coverage denominator is the held-out TEST set size.

This keeps evaluate_photoslam_largest_map.py as the formal TEST/NVS evaluator,
while summarize_tartanair_split_run.py separately reports TRAIN and TEST metrics.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "evaluate_photoslam_largest_map.py"
V1_MARKER = "render_metric_denominator_type"
V2_MARKER = "HELDOUT_TEST_FILTER_V2"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Patch anchor not found for {label}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text()
    if V2_MARKER in text:
        print(f"Already patched (v2): {PATH}")
        return 0

    # Install the denominator setup if a local v1 patch has not already done so.
    if V1_MARKER not in text:
        anchor = '''    selected_end = int(
        tracking_summary.get("selected_end_frame", largest_frame_ids[-1])
    )

'''
        block = '''    # Held-out NVS runs write exact test ids. Pose/ATE coverage still uses all
    # selected input frames, while image metrics use only the held-out test set.
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

    # v2: the runner may now render both train and test views. Keep the formal
    # largest-map evaluator test-only when a held-out split is present.
    anchor = '''    # Keep deterministic frame order even if CSV row order changes later.
    filtered_rows.sort(key=lambda r: int(r["frame_index"]))

'''
    block = '''    # HELDOUT_TEST_FILTER_V2: final_tracked_view_eval may contain both TRAIN and
    # TEST rows. Formal largest-map NVS metrics must remain TEST-only.
    if heldout_test_ids:
        heldout_test_set = set(heldout_test_ids)
        filtered_rows = [
            row for row in filtered_rows
            if int(row["frame_index"]) in heldout_test_set
        ]

'''
    text = replace_once(text, anchor, anchor + block, "held-out test row filter")

    PATH.write_text(text)
    print(f"Patched (v2): {PATH}")
    print("Largest-map PSNR/SSIM are TEST-only; pose/ATE coverage remains all-frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
