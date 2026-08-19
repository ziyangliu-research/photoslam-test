#!/usr/bin/env python3
"""Patch GaussianMapper shutdown handling for finite TartanAir evaluation runs.

Why this is needed
------------------
For short offline frame ranges, the runner can finish feeding images and call
ORB-SLAM3::Shutdown() before GaussianMapper has consumed the final local-mapping
operation. The original hasMetInitialMappingConditions() explicitly requires
SLAM to *not* be shut down, so a perfectly usable final ORB map can no longer
initialize the Gaussian map once shutdown begins.

The original code then enters the tail-optimization path with no Gaussian
keyframes, where trainForOneIteration() cannot advance the iteration counter and
the tail loop can run forever.

This patch makes two narrowly scoped changes:
  1. If the final ORB map already has at least Mapper.min_num_initial_map_kfs,
     allow initial Gaussian mapping after ORB-SLAM3 shutdown. This reuses the
     same initial-mapping code and the finalized ORB map; it does not lower the
     configured keyframe threshold.
  2. If the threshold still was not met and no Gaussian map exists, exit cleanly
     instead of entering the non-progressing tail loop.

Normal runs that initialize Gaussian mapping before shutdown are unchanged.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "gaussian_mapper.cpp"

OLD_CONDITION = '''    if (!pSLAM_->isShutDown() &&\n        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&\n        pSLAM_->getAtlas()->hasMappingOperation())\n        return true;\n'''

NEW_CONDITION = '''    // Offline finite-range evaluation may call ORB-SLAM3::Shutdown() before\n    // GaussianMapper consumes the last mapping operation. Once ORB shutdown is\n    // complete, the final Atlas map is stable, so allow initial Gaussian mapping\n    // from that final map as long as the configured KF threshold was reached.\n    if (pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&\n        (pSLAM_->getAtlas()->hasMappingOperation() || pSLAM_->isShutDown()))\n        return true;\n'''

OLD_GUARD = r'''    // Evaluation robustness guard: if SLAM shuts down before the minimum
    // number of keyframes required for initial Gaussian mapping is reached, the
    // original tail loop below cannot make progress because there is no training
    // viewpoint. Exit cleanly instead of looping forever. Normal initialized runs
    // are unchanged.
    if (!initial_mapped_) {
        std::cout << "[Gaussian Mapper]SLAM ended before initial Gaussian mapping was created; "
                  << "skipping tail optimization and Gaussian export." << std::endl;
        signalStop();
        return;
    }

'''

NEW_GUARD = r'''    // If the configured KF threshold still was not reached, there is no
    // trainable Gaussian scene. Do not enter the original tail loop because a
    // missing training viewpoint makes trainForOneIteration() roll the iteration
    // counter back and the tail loop cannot terminate.
    if (!initial_mapped_) {
        std::cout << "[Gaussian Mapper]Final ORB map did not satisfy the configured initial "
                  << "Gaussian-mapping conditions; skipping tail optimization and Gaussian export." << std::endl;
        signalStop();
        return;
    }

'''

SECOND_LOOP = "    // Second loop: Incremental gaussian mapping\n"


def main() -> int:
    text = PATH.read_text()
    changed = False

    # Upgrade the initial-condition logic. This is the actual short-sequence fix.
    if NEW_CONDITION not in text:
        if OLD_CONDITION not in text:
            raise RuntimeError(
                "Could not find the original hasMetInitialMappingConditions() block. "
                "Check src/gaussian_mapper.cpp before applying the patch."
            )
        text = text.replace(OLD_CONDITION, NEW_CONDITION, 1)
        changed = True

    # Upgrade an already-applied old guard, or insert the guard on a clean tree.
    if NEW_GUARD not in text:
        if OLD_GUARD in text:
            text = text.replace(OLD_GUARD, NEW_GUARD, 1)
            changed = True
        else:
            if SECOND_LOOP not in text:
                raise RuntimeError(f"Patch anchor not found in {PATH}")
            text = text.replace(SECOND_LOOP, NEW_GUARD + SECOND_LOOP, 1)
            changed = True

    if changed:
        PATH.write_text(text)
        print(f"Patched: {PATH}")
        print("Rebuild tartanair_stereo_eval before running again.")
    else:
        print(f"Already up to date: {PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
