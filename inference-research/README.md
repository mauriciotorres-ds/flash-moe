# Inference-Engine Research Project

An autoresearch study of local transformer inference, in the spirit of
Flash-MoE: build a reproducible baseline, run **40+ benchmarked optimization
experiments**, keep only what measurement validates, assemble the optimized
engine, validate transfer to a large-scale MoE, and ship a real-time
observability dashboard.

**Hardware target:** Apple M4 · 24 GB unified memory.
**Inference backend:** [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) (GGUF + Metal GPU).
**Phase-1 dev model (Small):** Qwen1.5-MoE-A2.7B — Q4_K_M GGUF (~8.8 GB), 60 experts/layer, top-4 active.
**Phase-2 validation model (Medium):** Qwen3.5-35B-A3B — Q4_K_M (~21 GB) and IQ2_M (~11 GB) GGUF.

> ### About the numbers currently on disk
> All results are **MEASURED** on an Apple M4 · 24 GB Mac (Small-model cycle run
> 2026-06-09). Every row is flagged `measured=true` and the dashboard shows a
> GREEN data-source banner. On the Small model: baseline **51.25 tok/s**, best
> config **exp014 (full Metal GPU offload + flash attention) at 98.96 tok/s**
> (**1.93× speedup**). 41 experiments run, **7 kept, 34 discarded**. See
> [reports/final_report.md](reports/final_report.md) for the full write-up
> including the cross-tier (Medium) validation.

---

## How to run it

### 0. Prerequisites

- macOS on Apple Silicon (M-series). Python 3.11+.
- ~10 GB free disk for the Small model (more for the Medium tiers).
- The models are **not** in the repo — you download them once (step 2).

### 1. Install

```bash
cd inference-research
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# llama-cpp-python must be built WITH Metal support (it is not in requirements.txt):
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir
```

### 2. Download the model weights

The engine looks for GGUF files under `~/models/<NAME>-GGUF/`. Download the
**Small** model first — it's all you need to run the full experiment cycle:

```bash
pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Small (Phase-1 dev model, ~8.8 GB) — required
hf download RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf \
  --include "*Q4_K_M*.gguf" \
  --local-dir ~/models/Qwen1.5-MoE-A2.7B-GGUF

# Medium (optional, for cross-tier validation)
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
  --include "*Q4_K_M*.gguf" --local-dir ~/models/Qwen3.5-35B-A3B-GGUF
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
  --include "*IQ2_M*.gguf"  --local-dir ~/models/Qwen3.5-35B-A3B-IQ2-GGUF
```

The directory names must match the ones in [`ireng/config.py`](ireng/config.py).
See [MSDS_MoE/project_spec.md](MSDS_MoE/project_spec.md) for alternate repos and
the (optional) Large tier.

### 3. Run the autoresearch experiment cycle

```bash
# Full run: 41 single-knob experiments on the Small model, keep/discard by measurement
python run_experiments.py --mode real

# Resume after an interruption (reads state.json, skips completed experiments)
python run_experiments.py --mode real --resume

# Run a subset by number
python run_experiments.py --mode real --start 11 --end 17

# No model handy? Dry-run the pipeline with synthetic numbers
python run_experiments.py --mode mock
```

Each experiment writes `experiments/expNNN.md`, appends to `results/*.csv|json`,
updates `results/best_config.json`, and checkpoints `state.json`.

### 4. Benchmark baseline vs. optimized

```bash
python benchmark_runner.py --engine both                  # Small model
python benchmark_runner.py --engine both --model medium     # Qwen3.5-35B Q4_K_M
python benchmark_runner.py --engine both --model medium-iq2 # Qwen3.5-35B IQ2_M
```

### 5. Regenerate plots & the final report

```bash
python plots/generate_plots.py
python reports/generate_report.py   # rebuilds reports/final_report.md from results/
```

`generate_report.py` is **data-driven**: every number in
[`reports/final_report.md`](reports/final_report.md) is read from `results/` and
`state.json`, so the report can't drift from the measurements. (The qualitative
narrative — why each optimization worked — is curated prose, but the numbers
inside it are injected from the data.)

### 6. Explore the dashboard

```bash
# Optional: seed clearly-labelled demo data so every view is populated
python seed_sample_data.py

streamlit run dashboard.py
```

> **Reproduce from a clean slate:** delete the prior outputs, then re-run the
> cycle (this re-measures everything and takes a while):
> ```bash
> rm -f results/*.csv results/*.json state.json failure_log.md experiments/exp*.md
> python run_experiments.py --mode real
> python plots/generate_plots.py && python reports/generate_report.py
> ```

## Documentation

The three components are documented separately so nothing is conflated:

- **[docs/ENGINE.md](docs/ENGINE.md)** — the baseline/configurable/optimized
  engines, every optimization knob, the generation lifecycle, quality gating.
- **[docs/DASHBOARD.md](docs/DASHBOARD.md)** — the six dashboard tabs, the
  data-source banner, live streaming and metrics.
- **[docs/PLOTS.md](docs/PLOTS.md)** — the six timeline charts and how they map
  to the data.

The data-driven **[reports/final_report.md](reports/final_report.md)** (rebuilt
by `reports/generate_report.py`) contains the methodology, the headline results,
the cross-tier validation, and the "What We Tried (And What Worked)" section.

## How it's organized

```
inference-research/
  run_experiments.py        # ▶ autoresearch loop (--mode real | mock)
  baseline_engine.py        # exp000 reference engine (GGUF, CPU-only, no flash-attn)
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
