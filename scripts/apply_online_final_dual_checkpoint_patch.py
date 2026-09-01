#!/usr/bin/env python3
"""Add ONLINE vs FINAL_TAIL evaluation checkpoints without changing Photo-SLAM optimization logic.

This patch is instrumentation only.

Original mapping/optimization flow remains:
  initial mapping -> incremental mapping -> tail Gaussian optimization -> shutdown save

Added observations:
  1. Immediately after the original incremental loop finishes, but BEFORE the
     original tail loop starts, record an ONLINE completion timestamp, active SH
     degree, and save an ONLINE PLY snapshot. No optimizer step, keyframe decision,
     densification rule, or tail-loop condition is changed.
  2. Let the original tail optimization and original _shutdown save run unchanged.
  3. After the mapper has completely exited, the evaluation runner first renders
     the untouched FINAL_TAIL state, then loads the saved ONLINE PLY, restores the
     ONLINE SH degree, and renders the same final-evaluable poses separately.

The ONLINE timestamp is captured before the added ONLINE PLY write, so ONLINE FPS
excludes checkpoint I/O, tail optimization, and offline metric rendering.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
MAPPER_H = ROOT / "include" / "gaussian_mapper.h"
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"
MAPPER_MARKER = "ONLINE checkpoint instrumentation (evaluation only)"
RUNNER_MARKER = "ONLINE vs FINAL_TAIL offline evaluation"
EVAL_SH_METHOD = "setShDegreeForEvaluation"


def patch_mapper_header() -> bool:
    text = MAPPER_H.read_text()
    if EVAL_SH_METHOD in text:
        return False

    anchor = '    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");\n'
    if anchor not in text:
        raise RuntimeError("Could not find GaussianMapper::loadPly declaration in include/gaussian_mapper.h")

    replacement = anchor + '''\n    // Evaluation-only helper. loadPly() restores all stored SH coefficients but\n    // sets active SH degree to the model maximum; an ONLINE checkpoint may have\n    // been using a lower degree at that exact iteration. Restore that degree for\n    // faithful checkpoint rendering. This is never called during mapping/training.\n    void setShDegreeForEvaluation(const int sh) { gaussians_->setShDegree(sh); }\n'''
    text = text.replace(anchor, replacement, 1)
    MAPPER_H.write_text(text)
    return True


def patch_mapper() -> bool:
    text = MAPPER.read_text()
    if MAPPER_MARKER in text:
        # Upgrade an earlier version of this instrumentation to record SH degree.
        old = '        online_meta << "online_iteration " << getIteration() << \'\\n\';\n        online_meta << "online_steady_clock_ns " << online_checkpoint_ns << \'\\n\';\n'
        new = old + '        online_meta << "online_sh_degree " << default_sh_ << \'\\n\';\n'
        if "online_sh_degree" not in text:
            if old not in text:
                raise RuntimeError("Existing ONLINE checkpoint block has an unexpected form")
            text = text.replace(old, new, 1)
            MAPPER.write_text(text)
            return True
        return False

    anchor = "    // Third loop: Tail gaussian optimization\n"
    if anchor not in text:
        raise RuntimeError("Could not find Photo-SLAM tail-optimization anchor in src/gaussian_mapper.cpp")

    block = r'''    // ONLINE checkpoint instrumentation (evaluation only).
    // This is exactly the boundary between the original incremental mapping loop
    // and the original tail-optimization loop. Capture time BEFORE checkpoint I/O.
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
    // Saving a snapshot does not modify Gaussian parameters. The original tail
    // optimization below is intentionally left untouched.
    savePly(result_dir_ / (std::to_string(getIteration()) + "_online") / "ply");

'''
    text = text.replace(anchor, block + anchor, 1)
    MAPPER.write_text(text)
    return True


def patch_runner() -> bool:
    text = RUNNER.read_text()
    if RUNNER_MARKER in text:
        # Upgrade earlier instrumentation to parse/restore online SH degree.
        changed = False
        if "long long online_sh_degree = -1;" not in text:
            old = '''    long long online_iteration = -1;
    long long online_steady_clock_ns = -1;
'''
            new = '''    long long online_iteration = -1;
    long long online_steady_clock_ns = -1;
    long long online_sh_degree = -1;
'''
            if old not in text:
                raise RuntimeError("Could not upgrade ONLINE metadata declarations")
            text = text.replace(old, new, 1)
            changed = True

        if 'else if (key == "online_sh_degree") online_sh_degree = value;' not in text:
            old = '''            if (key == "online_iteration") online_iteration = value;
            else if (key == "online_steady_clock_ns") online_steady_clock_ns = value;
'''
            new = '''            if (key == "online_iteration") online_iteration = value;
            else if (key == "online_steady_clock_ns") online_steady_clock_ns = value;
            else if (key == "online_sh_degree") online_sh_degree = value;
'''
            if old not in text:
                raise RuntimeError("Could not upgrade ONLINE metadata parser")
            text = text.replace(old, new, 1)
            changed = True

        if "setShDegreeForEvaluation" not in text:
            old = '''                        pGausMapper->loadPly(online_ply);
                        evaluateFinalTrackedViews(
'''
            new = '''                        pGausMapper->loadPly(online_ply);
                        if (online_sh_degree >= 0)
                            pGausMapper->setShDegreeForEvaluation(static_cast<int>(online_sh_degree));
                        evaluateFinalTrackedViews(
'''
            if old not in text:
                raise RuntimeError("Could not upgrade ONLINE PLY reload block")
            text = text.replace(old, new, 1)
            changed = True

        if changed:
            RUNNER.write_text(text)
        return changed

    # The all-view patch should already have been applied so train and test can
    # both be summarized from each checkpoint in one run.
    if "Split all-view final evaluation" not in text:
        raise RuntimeError(
            "Apply scripts/apply_tartanair_heldout_split_patch.py and "
            "scripts/apply_split_all_view_eval_patch.py before this patch."
        )

    # 1) Let the existing renderer choose its output subdirectory.
    old_sig = '''    torch::DeviceType device_type,
    const fs::path &output_dir)
{
    const fs::path eval_dir = output_dir / "final_tracked_view_eval";
'''
    new_sig = '''    torch::DeviceType device_type,
    const fs::path &output_dir,
    const std::string &eval_subdir)
{
    const fs::path eval_dir = output_dir / eval_subdir;
'''
    if old_sig not in text:
        raise RuntimeError("Could not patch evaluateFinalTrackedViews() signature/output directory")
    text = text.replace(old_sig, new_sig, 1)

    # 2) Read the mapper's exact pre-tail timestamp/SH degree after mapper exit.
    mapper_done_anchor = '''    const auto mapper_done = std::chrono::steady_clock::now();

    const double stream_wall_sec =
'''
    mapper_done_new = r'''    const auto mapper_done = std::chrono::steady_clock::now();

    // ONLINE vs FINAL_TAIL offline evaluation: recover the exact pre-tail
    // checkpoint metadata written by GaussianMapper. The timestamp was captured
    // before ONLINE PLY I/O, so the reported ONLINE FPS is compute-only.
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

    const double stream_wall_sec =
'''
    if mapper_done_anchor not in text:
        raise RuntimeError("Could not find mapper_done timing anchor")
    text = text.replace(mapper_done_anchor, mapper_done_new, 1)

    # 3) Add formal online timing fields. Existing final/pipeline timing is retained
    # for diagnostics but should not be used as the online FPS in paper tables.
    timing_anchor = '''        timing << "pipeline_until_gaussian_mapper_exit_wall_sec " << pipeline_wall_sec << '\n';
'''
    timing_new = '''        timing << "online_checkpoint_iteration " << online_iteration << '\n';
        timing << "online_checkpoint_sh_degree " << online_sh_degree << '\n';
        timing << "online_pipeline_wall_sec " << online_pipeline_wall_sec << '\n';
        timing << "online_pipeline_fps "
               << (online_pipeline_wall_sec > 0.0 ? static_cast<double>(tracking_records.size()) / online_pipeline_wall_sec : 0.0)
               << '\n';
        timing << "note_online_time_excludes_checkpoint_ply_write_tail_optimization_and_added_offline_rendering 1\n";
        timing << "pipeline_until_gaussian_mapper_exit_wall_sec " << pipeline_wall_sec << '\n';
'''
    if timing_anchor not in text:
        raise RuntimeError("Could not find timing_summary pipeline anchor")
    text = text.replace(timing_anchor, timing_new, 1)

    # 4) FINAL_TAIL first, then reload ONLINE PLY after official shutdown output is safe.
    call_old = '''                evaluateFinalTrackedViews(
                    frames,
                    tracking_records,
                    final_poses,
                    gaussian_keyframe_images,
                    pGausMapper,
                    device_type,
                    output_dir);
'''
    call_new = r'''                // FINAL_TAIL: untouched state produced by the original Photo-SLAM flow.
                evaluateFinalTrackedViews(
                    frames,
                    tracking_records,
                    final_poses,
                    gaussian_keyframe_images,
                    pGausMapper,
                    device_type,
                    output_dir,
                    "final_tracked_view_eval");

                // ONLINE: reload the pre-tail PLY only AFTER the official final
                // result has been saved and evaluated. This cannot affect mapping.
                if (online_iteration >= 0)
                {
                    const fs::path online_ply =
                        output_dir /
                        (std::to_string(online_iteration) + "_online") /
                        "ply" / "point_cloud" /
                        ("iteration_" + std::to_string(online_iteration)) /
                        "point_cloud.ply";
                    if (fs::exists(online_ply))
                    {
                        pGausMapper->loadPly(online_ply);
                        if (online_sh_degree >= 0)
                            pGausMapper->setShDegreeForEvaluation(static_cast<int>(online_sh_degree));
                        evaluateFinalTrackedViews(
                            frames,
                            tracking_records,
                            final_poses,
                            gaussian_keyframe_images,
                            pGausMapper,
                            device_type,
                            output_dir,
                            "online_tracked_view_eval");
                    }
                    else
                    {
                        std::cerr << "[ONLINE evaluation] checkpoint PLY not found: "
                                  << online_ply << std::endl;
                    }
                }
'''
    if call_old not in text:
        raise RuntimeError(
            "Could not find all-view evaluateFinalTrackedViews() call. "
            "Make sure apply_split_all_view_eval_patch.py was applied."
        )
    text = text.replace(call_old, call_new, 1)

    RUNNER.write_text(text)
    return True


def main() -> int:
    header_changed = patch_mapper_header()
    mapper_changed = patch_mapper()
    runner_changed = patch_runner()
    if header_changed or mapper_changed or runner_changed:
        print("Patched ONLINE/FINAL_TAIL evaluation instrumentation:")
        if header_changed:
            print(f"  {MAPPER_H}")
        if mapper_changed:
            print(f"  {MAPPER}")
        if runner_changed:
            print(f"  {RUNNER}")
        print("Original optimization loops/conditions are unchanged.")
        print("ONLINE SH degree is preserved for faithful PLY re-rendering.")
        print("Rebuild tartanair_stereo_eval before running.")
    else:
        print("ONLINE/FINAL_TAIL instrumentation already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
