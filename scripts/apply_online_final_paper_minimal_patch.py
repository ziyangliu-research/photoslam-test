#!/usr/bin/env python3
"""Minimal paper instrumentation for Photo-SLAM ONLINE vs FINAL_TAIL.

This patch intentionally changes NO tracking, mapping, optimization, keyframe,
densification, pruning, loop-closure, or tail-loop logic.

Required local patch order before this script:
  1. apply_gaussian_mapper_shutdown_guard.py
  2. apply_tartanair_heldout_split_patch.py
  3. apply_split_all_view_eval_patch.py

Added observations only:
  * At the exact boundary after the original incremental Gaussian mapping loop and
    before the original tail Gaussian optimization loop:
      - record iteration / steady-clock timestamp / active SH degree
      - save the current Gaussian PLY as *_online
  * Let the original Photo-SLAM tail loop and original *_shutdown save run unchanged.
  * After mapper exit, evaluate FINAL_TAIL first, then reload the ONLINE PLY and
    evaluate ONLINE using the SAME final-evaluable camera poses for both modes.
  * ONLINE FPS is measured to the pre-tail timestamp, before added PLY I/O, tail,
    and added offline rendering.

This deliberately does NOT save/recover alternative pose snapshots and does NOT
perform any post-hoc SE(3) recovery. It is intended for sequences that are fully
tracked and used in the paper comparison protocol.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
MAPPER_H = ROOT / "include" / "gaussian_mapper.h"
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"

MAPPER_MARKER = "PAPER MINIMAL ONLINE checkpoint (evaluation only)"
RUNNER_MARKER = "PAPER MINIMAL ONLINE vs FINAL_TAIL evaluation"


def patch_header() -> bool:
    text = MAPPER_H.read_text()
    if "setShDegreeForEvaluation" in text:
        return False
    anchor = '    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");\n'
    if anchor not in text:
        raise RuntimeError("GaussianMapper::loadPly declaration not found")
    text = text.replace(
        anchor,
        anchor + '\n    // Evaluation-only helper for re-rendering a saved ONLINE PLY.\n'
                 '    void setShDegreeForEvaluation(const int sh) { gaussians_->setShDegree(sh); }\n',
        1,
    )
    MAPPER_H.write_text(text)
    return True


def patch_mapper() -> bool:
    text = MAPPER.read_text()
    if MAPPER_MARKER in text:
        return False

    tail_anchor = "    // Third loop: Tail gaussian optimization\n"
    if tail_anchor not in text:
        raise RuntimeError("Photo-SLAM tail-loop anchor not found")

    block = r'''    // PAPER MINIMAL ONLINE checkpoint (evaluation only).
    // Exact boundary between the ORIGINAL incremental mapping loop and the
    // ORIGINAL tail Gaussian optimization loop. Capture time before added I/O.
    const auto online_checkpoint_tp = std::chrono::steady_clock::now();
    const long long online_checkpoint_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            online_checkpoint_tp.time_since_epoch()).count();
    {
        std::ofstream online_meta(result_dir_ / "online_checkpoint_metadata.txt");
        online_meta << "online_iteration " << getIteration() << '\n';
        online_meta << "online_steady_clock_ns " << online_checkpoint_ns << '\n';
        online_meta << "online_sh_degree " << default_sh_ << '\n';
    }
    // Snapshot only. No optimizer step or mapping decision is added here.
    savePly(result_dir_ / (std::to_string(getIteration()) + "_online") / "ply");

'''
    text = text.replace(tail_anchor, block + tail_anchor, 1)
    MAPPER.write_text(text)
    return True


def patch_runner() -> bool:
    text = RUNNER.read_text()
    if RUNNER_MARKER in text:
        return False
    if "Held-out NVS split" not in text:
        raise RuntimeError("Held-out split patch is missing")
    if "Split all-view final evaluation" not in text:
        raise RuntimeError("All-view evaluation patch is missing")

    changed = False

    # Make evaluation output directory selectable.
    if "const std::string &eval_subdir" not in text:
        pat = re.compile(
            r'(\s+torch::DeviceType device_type,\n\s+const fs::path &output_dir)\)\n\{\n'
            r'\s+const fs::path eval_dir = output_dir / "final_tracked_view_eval";'
        )
        m = pat.search(text)
        if not m:
            raise RuntimeError("evaluateFinalTrackedViews signature anchor not found")
        repl = (
            m.group(1) + ',\n    const std::string &eval_subdir)\n{\n'
            '    const fs::path eval_dir = output_dir / eval_subdir;'
        )
        text = text[:m.start()] + repl + text[m.end():]
        changed = True

    # Parse exact pre-tail checkpoint metadata after mapper exit.
    mapper_line = '    const auto mapper_done = std::chrono::steady_clock::now();\n'
    if mapper_line not in text:
        raise RuntimeError("mapper_done anchor not found")
    meta_block = r'''

    // PAPER MINIMAL ONLINE vs FINAL_TAIL evaluation.
    long long online_iteration = -1;
    long long online_steady_clock_ns = -1;
    long long online_sh_degree = -1;
    {
        std::ifstream online_meta(output_dir / "online_checkpoint_metadata.txt");
        std::string key;
        long long value = -1;
        while (online_meta >> key >> value)
        {
            if (key == "online_iteration") online_iteration = value;
            else if (key == "online_steady_clock_ns") online_steady_clock_ns = value;
            else if (key == "online_sh_degree") online_sh_degree = value;
        }
    }
    const long long stream_start_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            stream_start.time_since_epoch()).count();
    const double online_pipeline_wall_sec =
        (online_steady_clock_ns >= stream_start_ns)
            ? static_cast<double>(online_steady_clock_ns - stream_start_ns) / 1e9
            : -1.0;
'''
    text = text.replace(mapper_line, mapper_line + meta_block, 1)
    changed = True

    # Insert exact ONLINE timing fields before existing pipeline/tail timing.
    lines = text.splitlines(keepends=True)
    insert_idx = None
    for i, line in enumerate(lines):
        if 'timing << "pipeline_until_gaussian_mapper_exit_wall_sec "' in line:
            insert_idx = i
            break
    if insert_idx is None:
        raise RuntimeError("pipeline timing section not found")
    timing_block = '''        timing << "online_checkpoint_iteration " << online_iteration << '\\n';\n        timing << "online_checkpoint_sh_degree " << online_sh_degree << '\\n';\n        timing << "online_pipeline_wall_sec " << online_pipeline_wall_sec << '\\n';\n        timing << "online_pipeline_fps "\n               << (online_pipeline_wall_sec > 0.0 ? static_cast<double>(tracking_records.size()) / online_pipeline_wall_sec : 0.0)\n               << '\\n';\n        timing << "note_online_time_excludes_checkpoint_ply_write_tail_optimization_and_added_offline_rendering 1\\n";\n'''
    lines.insert(insert_idx, timing_block)
    text = ''.join(lines)
    changed = True

    # Replace the one all-view evaluation call with FINAL_TAIL then ONLINE.
    call_pat = re.compile(
        r'(?P<indent>\s*)evaluateFinalTrackedViews\(\n'
        r'(?P<body>\s+frames,\n\s+tracking_records,\n\s+final_poses,\n'
        r'\s+gaussian_keyframe_images,\n\s+pGausMapper,\n\s+device_type,\n\s+output_dir)\);'
    )
    m = call_pat.search(text)
    if not m:
        raise RuntimeError("all-view evaluateFinalTrackedViews call not found")
    indent = m.group('indent')
    replacement = f'''{indent}// FINAL_TAIL: original Photo-SLAM shutdown state.\n{indent}evaluateFinalTrackedViews(\n{indent}    frames,\n{indent}    tracking_records,\n{indent}    final_poses,\n{indent}    gaussian_keyframe_images,\n{indent}    pGausMapper,\n{indent}    device_type,\n{indent}    output_dir,\n{indent}    "final_tracked_view_eval");\n\n{indent}// ONLINE: same camera poses, only Gaussian state changes.\n{indent}if (online_iteration >= 0)\n{indent}{{\n{indent}    const fs::path online_ply =\n{indent}        output_dir /\n{indent}        (std::to_string(online_iteration) + "_online") /\n{indent}        "ply" / "point_cloud" /\n{indent}        ("iteration_" + std::to_string(online_iteration)) /\n{indent}        "point_cloud.ply";\n{indent}    if (fs::exists(online_ply))\n{indent}    {{\n{indent}        pGausMapper->loadPly(online_ply);\n{indent}        if (online_sh_degree >= 0)\n{indent}            pGausMapper->setShDegreeForEvaluation(static_cast<int>(online_sh_degree));\n{indent}        evaluateFinalTrackedViews(\n{indent}            frames,\n{indent}            tracking_records,\n{indent}            final_poses,\n{indent}            gaussian_keyframe_images,\n{indent}            pGausMapper,\n{indent}            device_type,\n{indent}            output_dir,\n{indent}            "online_tracked_view_eval");\n{indent}    }}\n{indent}    else\n{indent}        std::cerr << "[ONLINE evaluation] checkpoint PLY not found: " << online_ply << std::endl;\n{indent}}}'''
    text = text[:m.start()] + replacement + text[m.end():]
    changed = True

    if changed:
        RUNNER.write_text(text)
    return changed


def main() -> int:
    changed = []
    if patch_header(): changed.append(str(MAPPER_H))
    if patch_mapper(): changed.append(str(MAPPER))
    if patch_runner(): changed.append(str(RUNNER))

    if changed:
        print("Applied PAPER MINIMAL ONLINE/FINAL_TAIL instrumentation:")
        for p in changed:
            print("  ", p)
        print("No original tracking/mapping/optimization/tail-loop logic was changed.")
        print("Rebuild tartanair_stereo_eval before running.")
    else:
        print("PAPER MINIMAL instrumentation already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
