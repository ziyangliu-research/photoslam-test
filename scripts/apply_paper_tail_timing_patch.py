#!/usr/bin/env python3
"""Add measurement-only timing around Photo-SLAM's original tail optimization.

Apply AFTER apply_online_final_paper_minimal_patch.py.

This patch does not change the tail loop body, condition, iteration count, mapping,
tracking, densification, pruning, or any optimizer setting. It only records a
steady-clock timestamp immediately before and after the existing third-loop tail
optimization and writes the elapsed seconds to offline_tail_metadata.txt.

The timer starts after the added ONLINE checkpoint PLY has been saved, so the
reported offline tail time excludes evaluation/checkpoint I/O.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT / "src" / "gaussian_mapper.cpp"
MARKER = "PAPER TAIL timing instrumentation (measurement only)"


def main() -> int:
    text = MAPPER.read_text()
    if MARKER in text:
        print(f"PAPER tail timing already present: {MAPPER}")
        return 0

    if "PAPER MINIMAL ONLINE checkpoint (evaluation only)" not in text:
        raise RuntimeError(
            "PAPER MINIMAL instrumentation is missing. Run "
            "scripts/apply_online_final_paper_minimal_patch.py first."
        )

    old = '''    // Third loop: Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
'''

    new = '''    // Third loop: Tail gaussian optimization
    // PAPER TAIL timing instrumentation (measurement only).
    // The original loop below is byte-for-byte unchanged; only wall time around
    // it is observed. ONLINE checkpoint I/O has already finished before this.
    const auto paper_tail_start_tp = std::chrono::steady_clock::now();

    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    const auto paper_tail_end_tp = std::chrono::steady_clock::now();
    const double paper_tail_wall_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            paper_tail_end_tp - paper_tail_start_tp).count();
    {
        std::ofstream tail_meta(result_dir_ / "offline_tail_metadata.txt");
        tail_meta << std::fixed << std::setprecision(9);
        tail_meta << "offline_tail_optimization_wall_sec " << paper_tail_wall_sec << '\\n';
        tail_meta << "tail_start_iteration " << SLAM_stop_iter << '\\n';
        tail_meta << "tail_end_iteration " << getIteration() << '\\n';
        tail_meta << "note_excludes_online_checkpoint_io_shutdown_render_ply_and_metric_eval 1\\n";
    }

    // Save and clear
'''

    if old not in text:
        raise RuntimeError(
            "Original Photo-SLAM third-loop block was not found exactly. Refusing "
            "to patch an unexpected mapper implementation."
        )

    text = text.replace(old, new, 1)

    # std::fixed / std::setprecision are used only for the metadata text file.
    if "#include <iomanip>" not in text:
        anchor = '#include "include/gaussian_mapper.h"\n'
        if anchor not in text:
            raise RuntimeError("gaussian_mapper include anchor not found")
        text = text.replace(anchor, anchor + "#include <iomanip>\n", 1)

    MAPPER.write_text(text)
    print(f"Patched measurement-only tail timer: {MAPPER}")
    print("Original Photo-SLAM tail loop body/condition are unchanged.")
    print("Rebuild tartanair_stereo_eval before running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
