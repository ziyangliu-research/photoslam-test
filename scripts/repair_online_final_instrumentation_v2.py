#!/usr/bin/env python3
"""Repair/upgrade ONLINE vs FINAL_TAIL instrumentation without changing Photo-SLAM logic.

This is evaluation-only instrumentation.

Original execution remains:
  initial mapping -> incremental mapping -> tail Gaussian optimization -> shutdown

At the exact boundary before the original tail loop, this patch records:
  * ONLINE iteration / steady-clock timestamp / active SH degree
  * ONLINE Gaussian PLY snapshot
  * ONLINE per-frame Tcw snapshot reconstructed from ORB relative frame poses and
    the Gaussian scene's keyframe poses at that same boundary

The original tail loop is not edited. After shutdown, the runner:
  * evaluates FINAL_TAIL using the existing final pose collection
  * reloads the ONLINE PLY and evaluates ONLINE using the saved ONLINE frame poses
  * records exact ONLINE wall time/FPS before checkpoint I/O and tail refinement

The patch is designed to be applied after the held-out split + all-view patches and
uses structural anchors rather than one exact timing-line spelling.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
MAPPER_H = ROOT / "include" / "gaussian_mapper.h"
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"

MAPPER_MARKER = "ONLINE checkpoint instrumentation (evaluation only)"
POSE_MARKER = "ONLINE frame-pose snapshot (evaluation only)"
RUNNER_MARKER = "ONLINE vs FINAL_TAIL offline evaluation"
POSE_LOADER = "loadOnlineFramePoseSnapshot"


def ensure_header_helper() -> bool:
    text = MAPPER_H.read_text()
    if "setShDegreeForEvaluation" in text:
        return False
    anchor = '    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");\n'
    if anchor not in text:
        raise RuntimeError("GaussianMapper::loadPly declaration not found")
    text = text.replace(
        anchor,
        anchor
        + '\n    // Evaluation only: restore the active SH degree saved at an ONLINE checkpoint.\n'
          '    void setShDegreeForEvaluation(const int sh) { gaussians_->setShDegree(sh); }\n',
        1,
    )
    MAPPER_H.write_text(text)
    return True


def ensure_mapper_checkpoint_and_pose_snapshot() -> bool:
    text = MAPPER.read_text()
    changed = False

    if '#include <iomanip>' not in text:
        include_anchor = '#include "include/gaussian_mapper.h"\n'
        if include_anchor not in text:
            raise RuntimeError("gaussian_mapper include anchor not found")
        text = text.replace(include_anchor, include_anchor + '#include <iomanip>\n', 1)
        changed = True

    tail_anchor = "    // Third loop: Tail gaussian optimization\n"
    if tail_anchor not in text:
        raise RuntimeError("Photo-SLAM tail-loop anchor not found")

    # Install the basic checkpoint if an older local tree does not have it yet.
    if MAPPER_MARKER not in text:
        checkpoint = r'''    // ONLINE checkpoint instrumentation (evaluation only).
    // Exact boundary between original incremental mapping and original tail optimization.
    // Capture time BEFORE any added checkpoint I/O.
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

'''
        text = text.replace(tail_anchor, checkpoint + tail_anchor, 1)
        changed = True
    elif "online_sh_degree" not in text:
        meta_anchor = '        online_meta << "online_steady_clock_ns " << online_checkpoint_ns << \'\\n\';\n'
        if meta_anchor not in text:
            raise RuntimeError("Existing ONLINE metadata block has unexpected form")
        text = text.replace(
            meta_anchor,
            meta_anchor + '        online_meta << "online_sh_degree " << default_sh_ << \'\\n\';\n',
            1,
        )
        changed = True

    if POSE_MARKER not in text:
        # Put the pose snapshot immediately before the ONLINE PLY write when available;
        # otherwise immediately before the untouched tail loop.
        save_anchor_match = re.search(
            r'^\s*savePly\(result_dir_ / \(std::to_string\(getIteration\(\)\) \+ "_online"\) / "ply"\);\s*$',
            text,
            flags=re.MULTILINE,
        )
        insert_pos = save_anchor_match.start() if save_anchor_match else text.index(tail_anchor)
        pose_block = r'''    // ONLINE frame-pose snapshot (evaluation only).
    // Reconstruct every retained frame pose against the Gaussian scene's own
    // keyframe poses at this exact boundary. This avoids using a later ORB pose
    // graph to render an earlier Gaussian checkpoint.
    {
        std::ofstream online_pose_file(result_dir_ / "online_frame_poses.csv");
        online_pose_file << "timestamp";
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                online_pose_file << ",Tcw_" << r << c;
        online_pose_file << '\n';
        online_pose_file << std::scientific << std::setprecision(17);

        ORB_SLAM3::Tracking *tracker = pSLAM_->getTracker();
        auto lRit = tracker->mlpReferences.begin();
        auto lT = tracker->mlFrameTimes.begin();
        auto lbL = tracker->mlbLost.begin();
        for (auto lit = tracker->mlRelativeFramePoses.begin(),
                  lend = tracker->mlRelativeFramePoses.end();
             lit != lend && lRit != tracker->mlpReferences.end() &&
             lT != tracker->mlFrameTimes.end() && lbL != tracker->mlbLost.end();
             ++lit, ++lRit, ++lT, ++lbL)
        {
            if (*lbL)
                continue;

            ORB_SLAM3::KeyFrame *pKF = *lRit;
            if (!pKF)
                continue;

            Sophus::SE3f Trw;
            while (pKF && pKF->isBad())
            {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }
            if (!pKF)
                continue;

            auto gkf_it = scene_->keyframes().find(pKF->mnId);
            if (gkf_it == scene_->keyframes().end())
                continue;

            Trw = Trw * gkf_it->second->getPosef();
            const Sophus::SE3f Tcw = (*lit) * Trw;
            const Eigen::Matrix4f M = Tcw.matrix();

            online_pose_file << *lT;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    online_pose_file << ',' << M(r, c);
            online_pose_file << '\n';
        }
    }

'''
        text = text[:insert_pos] + pose_block + text[insert_pos:]
        changed = True

    # Ensure one ONLINE PLY save exists before the untouched tail loop.
    if '_online") / "ply"' not in text:
        pos = text.index(tail_anchor)
        text = text[:pos] + '    savePly(result_dir_ / (std::to_string(getIteration()) + "_online") / "ply");\n\n' + text[pos:]
        changed = True

    if changed:
        MAPPER.write_text(text)
    return changed


def ensure_pose_loader(text: str) -> tuple[str, bool]:
    if POSE_LOADER in text:
        return text, False

    anchor = "static void saveOrbKeyframeManifest("
    pos = text.find(anchor)
    if pos < 0:
        raise RuntimeError("saveOrbKeyframeManifest anchor not found")

    block = r'''// Evaluation-only loader for the exact pre-tail frame-pose snapshot.
static std::map<long long, Sophus::SE3f> loadOnlineFramePoseSnapshot(const fs::path &path)
{
    std::map<long long, Sophus::SE3f> result;
    std::ifstream in(path);
    if (!in.is_open())
        return result;

    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double timestamp = 0.0;
        if (!(iss >> timestamp)) continue;

        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        bool ok = true;
        for (int r = 0; r < 4 && ok; ++r)
            for (int c = 0; c < 4; ++c)
                if (!(iss >> M(r, c))) { ok = false; break; }
        if (!ok) continue;

        result[timestampKey(timestamp)] = Sophus::SE3f(
            Sophus::SO3f(M.block<3,3>(0,0)),
            M.block<3,1>(0,3));
    }
    return result;
}

'''
    return text[:pos] + block + text[pos:], True


def ensure_runner() -> bool:
    text = RUNNER.read_text()
    changed = False

    if "Held-out NVS split" not in text:
        raise RuntimeError(
            "Held-out split runner patch is missing. Run apply_tartanair_heldout_split_patch.py first."
        )
    if "Split all-view final evaluation" not in text:
        raise RuntimeError(
            "All-view evaluation patch is missing. Run apply_split_all_view_eval_patch.py first."
        )

    text, c = ensure_pose_loader(text)
    changed |= c

    # Make the evaluator output directory selectable.
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

    # Insert ONLINE metadata/timing recovery immediately after mapper_done.
    if RUNNER_MARKER not in text:
        mapper_line = '    const auto mapper_done = std::chrono::steady_clock::now();\n'
        if mapper_line not in text:
            raise RuntimeError("mapper_done anchor not found")
        meta_block = r'''

    // ONLINE vs FINAL_TAIL offline evaluation.
    // Metadata timestamp is captured before checkpoint I/O and original tail refinement.
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

    # Add exact ONLINE timing fields before the existing pipeline/tail timing fields.
    if 'timing << "online_pipeline_wall_sec "' not in text:
        lines = text.splitlines(keepends=True)
        insert_idx = None
        for i, line in enumerate(lines):
            if 'timing << "pipeline_' in line and 'wall_sec ' in line and 'pipeline_wall_sec' in line:
                insert_idx = i
                break
        if insert_idx is None:
            for i, line in enumerate(lines):
                if 'note_pipeline_time_' in line:
                    insert_idx = i
                    break
        if insert_idx is None:
            raise RuntimeError("Could not locate timing_summary pipeline section structurally")

        timing_block = '''        timing << "online_checkpoint_iteration " << online_iteration << '\\n';\n        timing << "online_checkpoint_sh_degree " << online_sh_degree << '\\n';\n        timing << "online_pipeline_wall_sec " << online_pipeline_wall_sec << '\\n';\n        timing << "online_pipeline_fps "\n               << (online_pipeline_wall_sec > 0.0 ? static_cast<double>(tracking_records.size()) / online_pipeline_wall_sec : 0.0)\n               << '\\n';\n        timing << "note_online_time_excludes_checkpoint_ply_write_tail_optimization_and_added_offline_rendering 1\\n";\n'''
        lines.insert(insert_idx, timing_block)
        text = ''.join(lines)
        changed = True

    # Replace the existing single final evaluation call with FINAL_TAIL + ONLINE.
    if '"online_tracked_view_eval"' not in text:
        call_pat = re.compile(
            r'(?P<indent>\s*)evaluateFinalTrackedViews\(\n'
            r'(?P<body>\s+frames,\n\s+tracking_records,\n\s+final_poses,\n'
            r'\s+gaussian_keyframe_images,\n\s+pGausMapper,\n\s+device_type,\n\s+output_dir)\);'
        )
        m = call_pat.search(text)
        if not m:
            raise RuntimeError("All-view evaluateFinalTrackedViews call not found")
        indent = m.group('indent')
        replacement = f'''{indent}// FINAL_TAIL: original Photo-SLAM shutdown state.\n{indent}evaluateFinalTrackedViews(\n{indent}    frames,\n{indent}    tracking_records,\n{indent}    final_poses,\n{indent}    gaussian_keyframe_images,\n{indent}    pGausMapper,\n{indent}    device_type,\n{indent}    output_dir,\n{indent}    "final_tracked_view_eval");\n\n{indent}// ONLINE: load the exact pre-tail Gaussian checkpoint and its matching\n{indent}// frame-pose snapshot. This happens only after the original tail/save finished.\n{indent}const auto online_poses = loadOnlineFramePoseSnapshot(\n{indent}    output_dir / "online_frame_poses.csv");\n{indent}if (online_iteration >= 0 && !online_poses.empty())\n{indent}{{\n{indent}    const fs::path online_ply =\n{indent}        output_dir /\n{indent}        (std::to_string(online_iteration) + "_online") /\n{indent}        "ply" / "point_cloud" /\n{indent}        ("iteration_" + std::to_string(online_iteration)) /\n{indent}        "point_cloud.ply";\n{indent}    if (fs::exists(online_ply))\n{indent}    {{\n{indent}        pGausMapper->loadPly(online_ply);\n{indent}        if (online_sh_degree >= 0)\n{indent}            pGausMapper->setShDegreeForEvaluation(static_cast<int>(online_sh_degree));\n{indent}        evaluateFinalTrackedViews(\n{indent}            frames,\n{indent}            tracking_records,\n{indent}            online_poses,\n{indent}            gaussian_keyframe_images,\n{indent}            pGausMapper,\n{indent}            device_type,\n{indent}            output_dir,\n{indent}            "online_tracked_view_eval");\n{indent}    }}\n{indent}    else\n{indent}        std::cerr << "[ONLINE evaluation] checkpoint PLY not found: " << online_ply << std::endl;\n{indent}}}\n{indent}else\n{indent}{{\n{indent}    std::cerr << "[ONLINE evaluation] exact pre-tail frame-pose snapshot missing/empty." << std::endl;\n{indent}}}'''
        text = text[:m.start()] + replacement + text[m.end():]
        changed = True

    if changed:
        RUNNER.write_text(text)
    return changed


def main() -> int:
    changed = []
    if ensure_header_helper():
        changed.append(str(MAPPER_H))
    if ensure_mapper_checkpoint_and_pose_snapshot():
        changed.append(str(MAPPER))
    if ensure_runner():
        changed.append(str(RUNNER))

    if changed:
        print("Repaired/upgraded ONLINE/FINAL_TAIL instrumentation:")
        for p in changed:
            print("  ", p)
        print("Original mapping/training/tail loop conditions are unchanged.")
        print("Rebuild tartanair_stereo_eval before benchmarking.")
    else:
        print("ONLINE/FINAL_TAIL v2 instrumentation already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
