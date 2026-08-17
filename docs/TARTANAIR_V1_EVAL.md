# TartanAir V1 Stereo Challenge evaluation

This evaluation path is for the original TartanAir V1 / CVPR Visual SLAM Stereo Challenge (`SE000`-`SE007`, etc.). It is intentionally separate from the TartanAir V2 config.

## V1 camera geometry

- Pinhole / rectified stereo
- Resolution: 640 x 480
- fx = 320, fy = 320
- cx = 320, cy = 240
- Baseline = 0.25 m

Use:

- `cfg/ORB_SLAM3/Stereo/TartanAir/TartanAirV1_Challenge.yaml`
- `cfg/gaussian_mapper/Stereo/TartanAir/TartanAirV1_Challenge_eval.yaml`

The challenge image filenames do not contain timestamps. `tartanair_stereo_eval` therefore uses a configurable synthetic timestamp rate; the default is 10 Hz. If a different rate is required, pass `--fps=<value>` and use the same value for pose evaluation.

## Build

```bash
cmake --build build --target tartanair_stereo_eval -j8
```

## SE000 run

```bash
CUDA_VISIBLE_DEVICES=0 ./bin/tartanair_stereo_eval \
  ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
  ./cfg/ORB_SLAM3/Stereo/TartanAir/TartanAirV1_Challenge.yaml \
  ./cfg/gaussian_mapper/Stereo/TartanAir/TartanAirV1_Challenge_eval.yaml \
  /home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo/SE000 \
  ./results/tartanair_v1_SE000_eval
```

The runner is headless by default. Add `viewer` only when a GUI viewer is wanted.

## Important output sets

### Per-input tracking state

- `frame_tracking_status.csv`
- `strict_success_frames.txt`

Every processed image is listed with its exact left/right source path, ORB-SLAM3 tracking state (`OK`, `RECENTLY_LOST`, `LOST`, etc.), whether the current frame actually had a pose, and a strict success flag.

### Keyframe identity

- `orb_keyframes.csv`
- `orb_keyframe_frames.txt`
- `gaussian_keyframes.csv`
- `gaussian_keyframe_frames.txt`

`gaussian_keyframes.csv` is the most direct manifest for the views actually stored in Photo-SLAM's Gaussian scene. It also points to Photo-SLAM's original final keyframe rendering and GT-image files in `<iteration>_shutdown/`.

### Final-map all-available-frame rendering

- `final_tracked_view_eval/metrics.csv`
- `final_tracked_view_eval/final_evaluable_frames.txt`
- `final_tracked_view_eval/summary.txt`
- `final_tracked_view_eval/rendered/000000_left.png`, etc.

The final Gaussian map is rendered after the full input sequence and Photo-SLAM tail optimization. The pose for each evaluated frame is reconstructed from ORB-SLAM3's final relative-frame/reference-keyframe bookkeeping, so final keyframe pose updates are reflected. Frames with no retained trajectory pose are not filled with GT, interpolation, or a previous pose.

`metrics.csv` contains frame index, exact source image, live tracking state, keyframe flag, PSNR, SSIM, and saved rendered-image path.

## ATE on matched valid poses

```bash
python3 scripts/evaluate_tartanair_v1_pose.py \
  --gt /home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SE000.txt \
  --trajectory_tum ./results/tartanair_v1_SE000_eval/CameraTrajectory_TUM.txt \
  --fps 10 \
  --out_dir ./results/tartanair_v1_SE000_eval/pose_eval
```

The main ATE is SE(3)-aligned with scale fixed to 1 because Photo-SLAM is stereo. The script reports pose coverage together with ATE and writes the exact matched frame indices.
