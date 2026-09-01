#!/usr/bin/env python3
"""Add ONLINE vs FINAL_TAIL evaluation checkpoints without changing Photo-SLAM optimization logic.

This patch is instrumentation only.

Original mapping/optimization flow remains:
  initial mapping -> incremental mapping -> tail Gaussian optimization -> shutdown save

Added observations:
  1. Immediately after the original incremental loop finishes, but BEFORE the
     original tail loop starts, record an ONLINE completion timestamp and save
     an ONLINE PLY snapshot. No optimizer step, keyframe decision, densification
     rule, or tail-loop condition is changed.
  2. Let the original tail optimization and original _shutdown save run unchanged.
  3. After the mapper has completely exited, the evaluation runner first renders
     the untouched FINAL_TAIL state, then loads the saved ONLINE PLY and renders
     the same final-evaluable poses into a separate directory.

The ONLINE timing timestamp is captured before the added ONLINE PLY write, so
ONLINE FPS excludes checkpoint I/O, tail optimization, and offline metric rendering.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"
MAPPER_MARKER = "ONLINE checkpoint instrumentation (evaluation only)"
RUNNER_MARKER = "ONLINE vs FINAL_TAIL offline evaluation"


def patch_mapper() -> bool:
    text = MAPPER.read_text()
    if MAPPER_MARKER in text:
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
        return False

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

    # 2) Read the mapper's exact pre-tail timestamp after mapper exit. steady_clock
    # values are comparable because producer and consumer are in the same process.
    mapper_done_anchor = '''    const auto mapper_done = std::chrono::steady_clock::now();

    const double stream_wall_sec =
'''
    mapper_done_new = r'''    const auto mapper_done = std::chrono::steady_clock::now();

    // ONLINE vs FINAL_TAIL offline evaluation: recover the exact pre-tail
    // checkpoint timestamp written by GaussianMapper. This timestamp was captured
    // before ONLINE PLY I/O, so the reported ONLINE FPS is compute-only.
    long long online_iteration = -1;
    long long online_steady_clock_ns = -1;
    {
        std::ifstream online_meta(output_dir / "online_checkpoint_metadata.txt");
        std::string key;
        long long value = -1;
        while (online_meta >> key >> value)
        {
            if (key == "online_iteration") online_iteration = value;
            else if (key == "online_steady_clock_ns") online_steady_clock_ns = value;
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

    # 4) Replace the existing single final render call with FINAL_TAIL first, then
    # load the saved ONLINE PLY and render the same poses. This happens only after
    # mapper exit and after the official _shutdown PLY has already been saved.
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
    mapper_changed = patch_mapper()
    runner_changed = patch_runner()
    if mapper_changed or runner_changed:
        print("Patched ONLINE/FINAL_TAIL evaluation instrumentation:")
        if mapper_changed:
            print(f"  {MAPPER}")
        if runner_changed:
            print(f"  {RUNNER}")
        print("Original optimization loops/conditions are unchanged.")
        print("Rebuild tartanair_stereo_eval before running.")
    else:
        print("ONLINE/FINAL_TAIL instrumentation already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
