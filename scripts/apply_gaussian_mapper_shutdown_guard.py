#!/usr/bin/env python3
"""Apply a minimal Photo-SLAM shutdown guard to src/gaussian_mapper.cpp.

Why this is needed
------------------
GaussianMapper::run() can enter its tail-optimization loop even when SLAM shuts
 down before the initial Gaussian map has been created. In that state there are
 no Gaussian keyframes, trainForOneIteration() cannot advance the iteration
 counter, and the tail loop never terminates.

This patch does NOT change runs in which Gaussian mapping initialized normally.
It only exits cleanly when initial_mapped_ is still false after the initial
mapping loop.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "gaussian_mapper.cpp"

MARKER = "[Gaussian Mapper]SLAM ended before initial Gaussian mapping was created"
NEEDLE = "    // Second loop: Incremental gaussian mapping\n"
GUARD = r'''    // Evaluation robustness guard: if SLAM shuts down before the minimum
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


def main() -> int:
    text = PATH.read_text()
    if MARKER in text:
        print(f"Already patched: {PATH}")
        return 0
    if NEEDLE not in text:
        raise RuntimeError(f"Patch anchor not found in {PATH}")
    text = text.replace(NEEDLE, GUARD + NEEDLE, 1)
    PATH.write_text(text)
    print(f"Patched: {PATH}")
    print("Rebuild tartanair_stereo_eval before running again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
