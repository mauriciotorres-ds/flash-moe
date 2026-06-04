#!/usr/bin/env python3
"""seed_sample_data.py — populate the project with SAMPLE (synthetic) results.

WHY THIS EXISTS
---------------
Real benchmark numbers can only be produced by running the model on your Mac
(Apple M4 / 24 GB, MPS). Until you do that, the dashboard, experiment explorer,
and plots would have nothing to show. This script seeds a complete, internally
consistent set of results so every view is demoable immediately.

EVERYTHING IT WRITES IS CLEARLY LABELLED SYNTHETIC:
  * every row has  measured = False  and  data_source = "SAMPLE_SYNTHETIC"
  * the dashboard shows a red "SAMPLE DATA" banner whenever it loads such rows
  * `state.json` carries  "data_source": "SAMPLE_SYNTHETIC"

To replace it with real measurements:
  rm -f results/*.csv results/*.json state.json failure_log.md experiments/exp*.md
  python run_experiments.py --mode real --device mps     # runs on your Mac

This respects the spec's "never fabricate measurements" rule: nothing here is
ever presented as a real result.
"""
from __future__ import annotations

import os
import sys

from ireng import storage as st
from ireng import runner


def main():
    print("Seeding SAMPLE (synthetic) data — clearly labelled, not real measurements.")

    def _wipe(path):
        """Best-effort clean slate: remove if allowed, else truncate to empty.
        (Some sandboxed mounts forbid unlink but allow truncate.)"""
        if not os.path.exists(path):
            return
        try:
            os.remove(path)
        except OSError:
            try:
                open(path, "w").close()
            except OSError:
                pass

    for path in (st.EXPERIMENTS_CSV, st.EXPERIMENTS_JSON, st.LEADERBOARD_CSV,
                 st.BENCH_HISTORY_CSV, st.BEST_CONFIG_JSON, st.FAILURE_LOG,
                 st.STATE_FILE):
        _wipe(path)
    for f in os.listdir(st.EXPERIMENTS):
        if f.startswith("exp") and f.endswith(".md"):
            _wipe(os.path.join(st.EXPERIMENTS, f))

    state = runner.run(mode="sample", resume=False, device="mps", commit=False)
    state["data_source"] = "SAMPLE_SYNTHETIC"
    state["status"] = "sample"
    st.write_state(state)
    print(f"Seeded {len(st.read_experiments())} experiment rows (SAMPLE_SYNTHETIC).")
    print("Run the dashboard:  streamlit run dashboard.py")
    print("Generate plots:     python plots/generate_plots.py")


if __name__ == "__main__":
    sys.exit(main())
