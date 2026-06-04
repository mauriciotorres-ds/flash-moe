#!/usr/bin/env python3
"""run_experiments.py — top-level entry point for the autoresearch loop.

This is the script you run on your Mac to execute the 40+ experiments and
produce REAL measurements:

  python run_experiments.py --mode real --device mps          # all experiments
  python run_experiments.py --mode real --only 4 6 13         # a subset
  python run_experiments.py --mode mock                       # pipeline dry-run

It is a thin wrapper around ireng.runner so the package's relative imports work
whether you launch from here or via `python -m ireng.runner`.
"""
from ireng.runner import main

if __name__ == "__main__":
    main()
