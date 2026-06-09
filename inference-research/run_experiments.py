#!/usr/bin/env python3
"""run_experiments.py — Autoresearch loop entry point.

Runs the 40 MoE expert-streaming experiments on Qwen1.5-MoE-A2.7B GGUF.

Usage:
  # Full real run (requires model downloaded to ~/models/Qwen1.5-MoE-A2.7B-GGUF/)
  python run_experiments.py --mode real

  # Resume from where you left off (reads state.json)
  python run_experiments.py --mode real --resume

  # Run only specific experiments
  python run_experiments.py --mode real --start 11 --end 17

  # Dry-run to verify the pipeline without a model
  python run_experiments.py --mode mock
"""
from __future__ import annotations

import argparse
import sys

from ireng import storage as st
from ireng.runner import ExperimentRunner


def main():
    ap = argparse.ArgumentParser(
        description="MoE inference autoresearch runner (GGUF + llama-cpp-python)"
    )
    ap.add_argument("--mode", choices=["real", "mock"], default="real",
                    help="real=measure on model, mock=synthetic pipeline test")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from state.json (skip completed experiments)")
    ap.add_argument("--start", type=int, default=None,
                    help="First experiment number to run")
    ap.add_argument("--end", type=int, default=None,
                    help="Last experiment number to run (inclusive)")
    ap.add_argument("--reset", action="store_true",
                    help="Reset state.json and start from scratch (dangerous)")
    args = ap.parse_args()

    if args.reset:
        ans = input("Reset state and ALL results? [y/N] ")
        if ans.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        st.write_state({
            "current_experiment": 0, "last_completed_experiment": -1,
            "baseline_tps": None, "best_tps": None,
            "best_config_dict": None, "status": "idle",
        })

    state = st.read_state()
    print(f"State: last_completed={state.get('last_completed_experiment', -1)}, "
          f"best_tps={state.get('best_tps')}")

    runner = ExperimentRunner(
        mode=args.mode,
        resume=args.resume or (args.start is None),
        start_exp=args.start,
        end_exp=args.end,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted. state.json is up to date — resume with --resume.")
        sys.exit(0)


if __name__ == "__main__":
    main()
