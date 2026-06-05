# Inference-Engine Research Project

An autoresearch study of local transformer inference, in the spirit of
Flash-MoE: build a reproducible baseline, run **40+ benchmarked optimization
experiments**, keep only what measurement validates, assemble the optimized
engine, validate transfer to a large-scale MoE, and ship a real-time
observability dashboard.

**Hardware target:** Apple M4 · 24 GB unified memory (MPS).
**Phase-1 dev model:** Qwen2.5-0.5B-Instruct.
**Phase-2 validation model:** Qwen3.5-397B-A17B (via the Flash-MoE Metal engine).

> ### About the numbers currently on disk
> All results are **MEASURED** on an Apple M4 · 24 GB Mac (run 2026-06-04/05).
> Every row is flagged `measured=true`, the dashboard shows a GREEN banner, and
> the plots carry no watermark. Baseline: **36.527 tok/s**. Best config:
> **exp041 (dynamic batching) at 56.612 tok/s** (1.55× speedup). 8 experiments
> kept, 34 discarded, 1 N/A (MoE-only). To reproduce from scratch:
> ```bash
> rm -f results/*.csv results/*.json state.json failure_log.md experiments/exp*.md
> python run_experiments.py --mode real --device mps
> python plots/generate_plots.py && python reports/generate_report.py
> ```

## Quick start

```bash
pip install -r requirements.txt

# 1) (optional) seed demo data so every view is populated
python seed_sample_data.py

# 2) explore
streamlit run dashboard.py
python plots/generate_plots.py

# 3) when ready, measure for real on your Mac
python run_experiments.py --mode real --device mps     # 40+ experiments
python benchmark_runner.py --engine both --device mps  # baseline vs optimized
python large_model/validate_transfer.py                # Phase-2 MoE transfer
```

## Documentation

The three components are documented separately so nothing is conflated:

- **[docs/ENGINE.md](docs/ENGINE.md)** — the baseline/configurable/optimized
  engines, every optimization knob, the generation lifecycle, quality gating.
- **[docs/DASHBOARD.md](docs/DASHBOARD.md)** — the six dashboard tabs, the
  data-source banner, live streaming and metrics.
- **[docs/PLOTS.md](docs/PLOTS.md)** — the six timeline charts and how they map
  to the data.

The auto-generated **[reports/final_report.md](reports/final_report.md)**
contains the "What We Tried (And What Worked)" section.

## How it's organized

```
inference-research/
  run_experiments.py        # ▶ autoresearch loop (real | mock | sample)
  baseline_engine.py        # reference engine (HF, fp32, eager)
  optimized_engine.py       # best validated config (from best_config.json)
  benchmark_runner.py       # reproducible suite + speedup
  dashboard.py              # streamlit observability platform
  seed_sample_data.py       # clearly-labelled demo data
  state.json                # session-continuity checkpoint
  failure_log.md            # every discarded experiment + why

  ireng/                    # the shared package (one source of truth)
    config.py               #   EngineConfig — all knobs
    engine.py               #   ConfigurableEngine — the only engine
    benchmark.py            #   suite runner + quality gating
    experiments.py          #   the 42 experiment definitions
    runner.py               #   autoresearch orchestration + persistence
    metrics.py              #   tok/s, TTFT, memory, CPU/GPU, KV cache
    prompts.py              #   5-category benchmark suite
    storage.py              #   all on-disk formats
    hardware.py             #   M4 target spec + host detection

  experiments/expNNN.md     # per-experiment write-ups
  results/                  # experiments.csv/json, leaderboard, best_config, history
  reports/                  # final_report.md + generator
  plots/                    # PNG charts + generator
  large_model/              # 397B MoE transfer-validation harness
  dashboard_logs/           # optional per-run logs
```

## The autoresearch method

Each experiment is a one-knob delta on the current-best config (ablations delta
the baseline). The runner benchmarks it on the fixed 5-category prompt suite,
compares to the current best on **measured** tok/s, and keeps it only if it
improves throughput/latency/memory **without** degrading output quality. Wins
accumulate; failures are logged. Everything is written to disk after every
experiment and `state.json` lets the run resume after any interruption — so the
project survives a closed laptop or a dropped connection without re-running
work.

## Session continuity

`state.json` tracks phase, current/last experiment, baseline and best tok/s, and
status. On restart the runner reads it, skips completed experiments, and cleanly
re-runs any experiment that was mid-flight when interrupted.
