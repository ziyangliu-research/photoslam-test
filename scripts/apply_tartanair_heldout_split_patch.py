#!/usr/bin/env python3
"""Patch tartanair_stereo_eval.cpp with held-out test-view support.

The split is designed for fair novel-view-synthesis evaluation:
  * every Nth frame at the chosen offset is a held-out TEST frame;
  * TEST frames still run TrackStereo and therefore obtain an estimated pose;
  * TEST frames run with ORB-SLAM3 Tracking::InformOnlyTracking(true), so they
    cannot be inserted as new keyframes and therefore do not enter Photo-SLAM's
    Gaussian mapping/training path;
  * TRAIN frames use normal tracking/mapping;
  * final PSNR/SSIM rendering is restricted to held-out TEST frames;
  * ATE remains available from the normal final trajectory / largest-map script.

Default intended experiment: --test-every=5 --test-offset=4, i.e.
4,9,14,... are test views (20%) and the other frames are training/mapping views.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples" / "tartanair_stereo_eval.cpp"
MARKER = "Held-out NVS split"


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

    text = replace_once(
        text,
        ' *   - compute-limited timing metadata with no realtime playback sleep\n',
        ' *   - compute-limited timing metadata with no realtime playback sleep\n'
        ' *   - Held-out NVS split: test frames estimate pose but cannot create ORB/Gaussian keyframes\n',
        "file header",
    )

    text = replace_once(
        text,
        '    bool strict_success = false;\n};\n',
        '    bool strict_success = false;\n'
        '    bool is_test = false;\n'
        '};\n',
        "TrackingRecord",
    )

    text = replace_once(
        text,
        '    csv << "frame_index,timestamp,left_image,right_image,tracking_state,tracking_state_name,pose_set,strict_success\\n";\n',
        '    csv << "frame_index,timestamp,left_image,right_image,tracking_state,tracking_state_name,pose_set,strict_success,split\\n";\n',
        "tracking CSV header",
    )

    text = replace_once(
        text,
        "            << (r.pose_set ? 1 : 0) << ','\n"
        "            << (r.strict_success ? 1 : 0) << '\\n';\n",
        "            << (r.pose_set ? 1 : 0) << ','\n"
        "            << (r.strict_success ? 1 : 0) << ','\n"
        "            << (r.is_test ? \"test\" : \"train\") << '\\n';\n",
        "tracking CSV row",
    )

    text = replace_once(
        text,
        '            << " [viewer] [--fps=10.0] [--start=0] [--end=N] [--num-frames=N] [--skip-final-eval]"\n',
        '            << " [viewer] [--fps=10.0] [--start=0] [--end=N] [--num-frames=N] [--skip-final-eval]"\n'
        '            << " [--test-every=N] [--test-offset=N]"\n',
        "usage",
    )

    text = replace_once(
        text,
        '    bool skip_final_eval = false;\n'
        '    double fps = 10.0;\n',
        '    bool skip_final_eval = false;\n'
        '    int test_every = 0;\n'
        '    int test_offset = 0;\n'
        '    double fps = 10.0;\n',
        "split variables",
    )

    text = replace_once(
        text,
        '        else if (arg == "--skip-final-eval") skip_final_eval = true;\n',
        '        else if (arg == "--skip-final-eval") skip_final_eval = true;\n'
        '        else if (arg.rfind("--test-every=", 0) == 0) test_every = std::stoi(arg.substr(13));\n'
        '        else if (arg.rfind("--test-offset=", 0) == 0) test_offset = std::stoi(arg.substr(14));\n',
        "argument parser",
    )

    # Insert split construction immediately after the selected frame vector exists.
    anchor = '    std::cout << "TartanAir stereo sequence: " << sequence_root << std::endl;\n'
    split_block = r'''    // Held-out NVS split. With --test-every=5 --test-offset=4,
    // frame ids 4,9,14,... are test views. They are still tracked for pose,
    // but Tracking::InformOnlyTracking(true) prevents new keyframe insertion.
    if (test_every < 0)
        throw std::runtime_error("--test-every must be >= 0");
    if (test_every > 0 && (test_offset < 0 || test_offset >= test_every))
        throw std::runtime_error("--test-offset must satisfy 0 <= offset < test-every");

    const bool heldout_split_enabled = test_every > 0;
    auto is_test_frame = [test_every, test_offset](std::size_t frame_index) {
        return test_every > 0 &&
               static_cast<int>(frame_index % static_cast<std::size_t>(test_every)) == test_offset;
    };

    std::vector<InputFrame> evaluation_frames;
    evaluation_frames.reserve(frames.size());
    std::ofstream train_ids(output_dir / "train_frame_ids.txt");
    std::ofstream test_ids(output_dir / "test_frame_ids.txt");
    std::size_t train_frame_count = 0;
    std::size_t test_frame_count = 0;
    for (const auto &frame : frames)
    {
        if (heldout_split_enabled && is_test_frame(frame.frame_index))
        {
            test_ids << frame.frame_index << '\n';
            evaluation_frames.push_back(frame);
            ++test_frame_count;
        }
        else
        {
            train_ids << frame.frame_index << '\n';
            ++train_frame_count;
        }
    }
    if (!heldout_split_enabled)
        evaluation_frames = frames;

'''
    text = replace_once(text, anchor, split_block + anchor, "split construction")

    text = replace_once(
        text,
        '    std::cout << "Offline final tracked-view evaluation: " << (skip_final_eval ? "disabled" : "enabled") << std::endl;\n',
        '    std::cout << "Offline final tracked-view evaluation: " << (skip_final_eval ? "disabled" : "enabled") << std::endl;\n'
        '    if (heldout_split_enabled)\n'
        '        std::cout << "Held-out NVS split: every " << test_every << " frames, offset " << test_offset\n'
        '                  << " -> train=" << train_frame_count << ", test=" << test_frame_count << std::endl;\n',
        "split console summary",
    )

    text = replace_once(
        text,
        '        pSLAM->TrackStereo(\n',
        '        const bool heldout_test_frame = heldout_split_enabled && is_test_frame(frame.frame_index);\n'
        '        // Keep pose estimation, but prevent this held-out frame from becoming\n'
        '        // an ORB keyframe / Photo-SLAM Gaussian training view.\n'
        '        pSLAM->getTracker()->InformOnlyTracking(heldout_test_frame);\n\n'
        '        pSLAM->TrackStereo(\n',
        "per-frame tracking mode",
    )

    text = replace_once(
        text,
        '        tracking_records.push_back(record);\n',
        '        record.is_test = heldout_test_frame;\n'
        '        tracking_records.push_back(record);\n',
        "record split",
    )

    text = replace_once(
        text,
        '    const auto stream_end = std::chrono::steady_clock::now();\n',
        '    // Restore normal mode before shutdown; the last selected frame may be held out.\n'
        '    pSLAM->getTracker()->InformOnlyTracking(false);\n'
        '    const auto stream_end = std::chrono::steady_clock::now();\n',
        "restore tracking mode",
    )

    text = replace_once(
        text,
        '                evaluateFinalTrackedViews(\n'
        '                    frames,\n',
        '                evaluateFinalTrackedViews(\n'
        '                    evaluation_frames,\n',
        "held-out final evaluation",
    )

    # Add split metadata to timing and tracking summaries.
    text = replace_once(
        text,
        '        timing << "processed_frames " << tracking_records.size() << \'\\n\';\n',
        '        timing << "processed_frames " << tracking_records.size() << \'\\n\';\n'
        '        timing << "heldout_split_enabled " << (heldout_split_enabled ? 1 : 0) << \'\\n\';\n'
        '        timing << "train_frames " << train_frame_count << \'\\n\';\n'
        '        timing << "test_frames " << test_frame_count << \'\\n\';\n'
        '        timing << "test_every " << test_every << \'\\n\';\n'
        '        timing << "test_offset " << test_offset << \'\\n\';\n',
        "timing split metadata",
    )

    text = replace_once(
        text,
        '    summary << "processed_frames " << tracking_records.size() << \'\\n\';\n',
        '    summary << "processed_frames " << tracking_records.size() << \'\\n\';\n'
        '    summary << "heldout_split_enabled " << (heldout_split_enabled ? 1 : 0) << \'\\n\';\n'
        '    summary << "train_frames " << train_frame_count << \'\\n\';\n'
        '    summary << "test_frames " << test_frame_count << \'\\n\';\n'
        '    summary << "test_every " << test_every << \'\\n\';\n'
        '    summary << "test_offset " << test_offset << \'\\n\';\n',
        "tracking split metadata",
    )

    PATH.write_text(text)
    print(f"Patched: {PATH}")
    print("Held-out NVS split enabled via --test-every=N --test-offset=N")
    print("Rebuild tartanair_stereo_eval before running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
