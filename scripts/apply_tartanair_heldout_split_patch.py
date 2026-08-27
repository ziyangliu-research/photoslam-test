#!/usr/bin/env python3
"""Patch Photo-SLAM/ORB-SLAM3 for held-out TartanAir NVS evaluation.

Goal
----
With --test-every=5 --test-offset=4, frames 4,9,14,... are held out:
  * they still run the NORMAL stereo tracking path and can obtain an estimated pose;
  * they cannot initialize an ORB map or create a new ORB keyframe;
  * therefore their images cannot enter Photo-SLAM Gaussian mapping/training;
  * final PSNR/SSIM rendering is restricted to those held-out frames.

Important implementation detail
-------------------------------
We intentionally DO NOT use Tracking::InformOnlyTracking(true), because ORB-SLAM3's
localization-only mode changes more than keyframe insertion (it changes the normal
tracking path and VO fallback behavior). Instead this patch adds a dedicated
SuppressKeyFrameInsertion() flag that only blocks StereoInitialization() and
NeedNewKeyFrame() for the selected held-out frame. Normal pose tracking otherwise
runs unchanged.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "tartanair_stereo_eval.cpp"
TRACKING_H = ROOT / "ORB-SLAM3" / "include" / "Tracking.h"
TRACKING_CC = ROOT / "ORB-SLAM3" / "src" / "Tracking.cc"


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"Patch anchor not found for {label}")
    return text.replace(old, new, 1), True


def patch_tracking_h() -> bool:
    text = TRACKING_H.read_text()
    changed = False

    text, c = replace_once(
        text,
        '    // Use this function if you have deactivated local mapping and you only want to localize the camera.\n'
        '    void InformOnlyTracking(const bool &flag);\n',
        '    // Use this function if you have deactivated local mapping and you only want to localize the camera.\n'
        '    void InformOnlyTracking(const bool &flag);\n\n'
        '    // Evaluation-only control: keep the normal tracking path but prevent the\n'
        '    // current frame from initializing/inserting mapping keyframes.\n'
        '    void SuppressKeyFrameInsertion(const bool &flag);\n',
        "Tracking.h setter declaration",
    )
    changed |= c

    text, c = replace_once(
        text,
        '    bool mbMapUpdated;\n',
        '    bool mbMapUpdated;\n'
        '    bool mbSuppressKeyFrameInsertion = false;\n',
        "Tracking.h suppression flag",
    )
    changed |= c

    if changed:
        TRACKING_H.write_text(text)
    return changed


def patch_tracking_cc() -> bool:
    text = TRACKING_CC.read_text()
    changed = False

    text, c = replace_once(
        text,
        'void Tracking::StereoInitialization()\n{\n',
        'void Tracking::StereoInitialization()\n{\n'
        '    // Held-out evaluation frames must not seed a persistent ORB map.\n'
        '    if(mbSuppressKeyFrameInsertion)\n'
        '        return;\n\n',
        "StereoInitialization guard",
    )
    changed |= c

    text, c = replace_once(
        text,
        'bool Tracking::NeedNewKeyFrame()\n{\n',
        'bool Tracking::NeedNewKeyFrame()\n{\n'
        '    // Held-out evaluation frames keep normal pose tracking but are forbidden\n'
        '    // from becoming persistent ORB/Photo-SLAM mapping keyframes.\n'
        '    if(mbSuppressKeyFrameInsertion)\n'
        '        return false;\n\n',
        "NeedNewKeyFrame guard",
    )
    changed |= c

    setter_old = '''void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}
'''
    setter_new = '''void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::SuppressKeyFrameInsertion(const bool &flag)
{
    mbSuppressKeyFrameInsertion = flag;
}
'''
    text, c = replace_once(text, setter_old, setter_new, "suppression setter")
    changed |= c

    if changed:
        TRACKING_CC.write_text(text)
    return changed


def patch_runner() -> bool:
    text = RUNNER.read_text()
    changed = False

    # Upgrade an earlier local v1 patch if it was applied before this script update.
    if 'pSLAM->getTracker()->InformOnlyTracking(heldout_test_frame);' in text:
        text = text.replace(
            'pSLAM->getTracker()->InformOnlyTracking(heldout_test_frame);',
            'pSLAM->getTracker()->SuppressKeyFrameInsertion(heldout_test_frame);',
        )
        text = text.replace(
            'pSLAM->getTracker()->InformOnlyTracking(false);',
            'pSLAM->getTracker()->SuppressKeyFrameInsertion(false);',
        )
        changed = True

    if "Held-out NVS split" not in text:
        replacements = [
            (
                ' *   - compute-limited timing metadata with no realtime playback sleep\n',
                ' *   - compute-limited timing metadata with no realtime playback sleep\n'
                ' *   - Held-out NVS split: test frames estimate pose but cannot create ORB/Gaussian keyframes\n',
                "file header",
            ),
            (
                '    bool strict_success = false;\n};\n',
                '    bool strict_success = false;\n'
                '    bool is_test = false;\n'
                '};\n',
                "TrackingRecord",
            ),
            (
                '    csv << "frame_index,timestamp,left_image,right_image,tracking_state,tracking_state_name,pose_set,strict_success\\n";\n',
                '    csv << "frame_index,timestamp,left_image,right_image,tracking_state,tracking_state_name,pose_set,strict_success,split\\n";\n',
                "tracking CSV header",
            ),
            (
                "            << (r.pose_set ? 1 : 0) << ','\n"
                "            << (r.strict_success ? 1 : 0) << '\\n';\n",
                "            << (r.pose_set ? 1 : 0) << ','\n"
                "            << (r.strict_success ? 1 : 0) << ','\n"
                "            << (r.is_test ? \"test\" : \"train\") << '\\n';\n",
                "tracking CSV row",
            ),
            (
                '            << " [viewer] [--fps=10.0] [--start=0] [--end=N] [--num-frames=N] [--skip-final-eval]"\n',
                '            << " [viewer] [--fps=10.0] [--start=0] [--end=N] [--num-frames=N] [--skip-final-eval]"\n'
                '            << " [--test-every=N] [--test-offset=N]"\n',
                "usage",
            ),
            (
                '    bool skip_final_eval = false;\n'
                '    double fps = 10.0;\n',
                '    bool skip_final_eval = false;\n'
                '    int test_every = 0;\n'
                '    int test_offset = 0;\n'
                '    double fps = 10.0;\n',
                "split variables",
            ),
            (
                '        else if (arg == "--skip-final-eval") skip_final_eval = true;\n',
                '        else if (arg == "--skip-final-eval") skip_final_eval = true;\n'
                '        else if (arg.rfind("--test-every=", 0) == 0) test_every = std::stoi(arg.substr(13));\n'
                '        else if (arg.rfind("--test-offset=", 0) == 0) test_offset = std::stoi(arg.substr(14));\n',
                "argument parser",
            ),
        ]
        for old, new, label in replacements:
            text, c = replace_once(text, old, new, label)
            changed |= c

        anchor = '    std::cout << "TartanAir stereo sequence: " << sequence_root << std::endl;\n'
        split_block = r'''    // Held-out NVS split. With --test-every=5 --test-offset=4,
    // frame ids 4,9,14,... are test views. They are tracked normally for pose,
    // but cannot initialize or insert persistent mapping keyframes.
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
        text, c = replace_once(text, anchor, split_block + anchor, "split construction")
        changed |= c

        more = [
            (
                '    std::cout << "Offline final tracked-view evaluation: " << (skip_final_eval ? "disabled" : "enabled") << std::endl;\n',
                '    std::cout << "Offline final tracked-view evaluation: " << (skip_final_eval ? "disabled" : "enabled") << std::endl;\n'
                '    if (heldout_split_enabled)\n'
                '        std::cout << "Held-out NVS split: every " << test_every << " frames, offset " << test_offset\n'
                '                  << " -> train=" << train_frame_count << ", test=" << test_frame_count << std::endl;\n',
                "split console summary",
            ),
            (
                '        pSLAM->TrackStereo(\n',
                '        const bool heldout_test_frame = heldout_split_enabled && is_test_frame(frame.frame_index);\n'
                '        // Suppress persistent map/keyframe insertion only; normal pose tracking remains active.\n'
                '        pSLAM->getTracker()->SuppressKeyFrameInsertion(heldout_test_frame);\n\n'
                '        pSLAM->TrackStereo(\n',
                "per-frame suppression",
            ),
            (
                '        tracking_records.push_back(record);\n',
                '        record.is_test = heldout_test_frame;\n'
                '        tracking_records.push_back(record);\n',
                "record split",
            ),
            (
                '    const auto stream_end = std::chrono::steady_clock::now();\n',
                '    // The last selected frame may be held out; restore normal behavior before shutdown.\n'
                '    pSLAM->getTracker()->SuppressKeyFrameInsertion(false);\n'
                '    const auto stream_end = std::chrono::steady_clock::now();\n',
                "restore suppression",
            ),
            (
                '                evaluateFinalTrackedViews(\n'
                '                    frames,\n',
                '                evaluateFinalTrackedViews(\n'
                '                    evaluation_frames,\n',
                "held-out final evaluation",
            ),
            (
                '        timing << "processed_frames " << tracking_records.size() << \'\\n\';\n',
                '        timing << "processed_frames " << tracking_records.size() << \'\\n\';\n'
                '        timing << "heldout_split_enabled " << (heldout_split_enabled ? 1 : 0) << \'\\n\';\n'
                '        timing << "train_frames " << train_frame_count << \'\\n\';\n'
                '        timing << "test_frames " << test_frame_count << \'\\n\';\n'
                '        timing << "test_every " << test_every << \'\\n\';\n'
                '        timing << "test_offset " << test_offset << \'\\n\';\n',
                "timing split metadata",
            ),
            (
                '    summary << "processed_frames " << tracking_records.size() << \'\\n\';\n',
                '    summary << "processed_frames " << tracking_records.size() << \'\\n\';\n'
                '    summary << "heldout_split_enabled " << (heldout_split_enabled ? 1 : 0) << \'\\n\';\n'
                '    summary << "train_frames " << train_frame_count << \'\\n\';\n'
                '    summary << "test_frames " << test_frame_count << \'\\n\';\n'
                '    summary << "test_every " << test_every << \'\\n\';\n'
                '    summary << "test_offset " << test_offset << \'\\n\';\n',
                "tracking split metadata",
            ),
        ]
        for old, new, label in more:
            text, c = replace_once(text, old, new, label)
            changed |= c

    if changed:
        RUNNER.write_text(text)
    return changed


def main() -> int:
    changed_h = patch_tracking_h()
    changed_cc = patch_tracking_cc()
    changed_runner = patch_runner()

    if changed_h or changed_cc or changed_runner:
        print("Patched held-out NVS support:")
        if changed_h:
            print(f"  {TRACKING_H}")
        if changed_cc:
            print(f"  {TRACKING_CC}")
        if changed_runner:
            print(f"  {RUNNER}")
        print("Rebuild tartanair_stereo_eval before running.")
    else:
        print("Held-out NVS patch already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
