# The Dashboard — How It Works

This document explains the Streamlit observability dashboard on its own. For the
engine see [ENGINE.md](ENGINE.md); for the charts see [PLOTS.md](PLOTS.md).

The dashboard is a **Phase-2 deliverable** — per the spec it is meant to be
used *after* the optimization cycle, to demonstrate and explore what the
experiments produced. It reads the same on-disk results the runner writes, so it
reflects SAMPLE data now and real measurements once you run on your Mac.

## Launch

```bash
pip install -r requirements.txt
python seed_sample_data.py        # optional: demo data so every tab is populated
streamlit run dashboard.py
```

The dashboard opens in your browser. A header shows the **hardware target**
(Apple M4 · 24 GB), the detected host device, the torch version, and the number
of experiments on disk.

## The data-source banner (read this first)

Right under the header the dashboard prints a banner stating where the loaded
numbers came from:

- 🔴 **SAMPLE DATA** — synthetic placeholders from `seed_sample_data.py`. Shown
  in red so a demo can never be mistaken for a measurement.
- 🟠 **Mock dry-run** — synthetic numbers from a pipeline test.
- 🟢 **Measured data** — real results from running the model on the host.
- ℹ️ **No data yet** — nothing has been run.

This banner is the dashboard's honesty contract: it never presents synthetic
data as real.

## The six tabs

**1. Live Playground.** Enter a system prompt and prompt, pick the model
(Qwen2.5-0.5B-Instruct or Qwen3.5-397B-A17B), the engine (Baseline or
Optimized), the device, and the generation parameters (max tokens, temperature,
top-p, top-k, seed). Press **Run** and the output **streams live** while five
metrics update in real time: tokens, tok/s, TTFT, elapsed, context length. When
generation finishes, a second row shows peak/current memory, CPU%, GPU%
(`n/a (MPS)` on Apple), and KV-cache size. Each run can be logged to
`dashboard_logs/`. Selecting the 397B model explains that it runs through the
Metal engine, not this Python path.

**2. Compare.** Shows the latest baseline-vs-optimized comparison written by
`benchmark_runner.py --engine both`, including the computed
`speedup = optimized_tps / baseline_tps`.

**3. Experiment Explorer.** A sortable, filterable table of all 42 experiments.
Filter by category or decision (keep/discard/n-a), sort by tok/s, latency, or
memory. Selecting an experiment renders its full `experiments/expNNN.md`
write-up — hypothesis, rationale, config delta, results, decision, lessons.

**4. Optimization Timeline.** Interactive line charts of tok/s, speedup-vs-
baseline, latency, and memory across experiment number, plus the static PNGs
from `plots/` if you have generated them.

**5. Diff Viewer.** A table diff of the baseline config versus the optimized
config — every knob that changed, with its before/after value — sourced from
`best_config.json`.

**6. MoE Visualization.** For the 397B model: the optimization **transfer map**
(small-model win → MoE-engine equivalent) and the results of
`large_model/validate_transfer.py`. Per-expert routing heatmaps require
instrumenting the Metal engine to emit activation counts; until then the tab
states that limitation honestly rather than showing empty charts.

## How it stays in sync with the engine

The dashboard imports the same `ireng` package and the same
`baseline_engine` / `optimized_engine` wrappers used everywhere else. Live
generation in the Playground uses `ConfigurableEngine.stream`, so the tok/s and
TTFT you watch are produced by the exact code path the benchmarks measure —
there is no separate "dashboard engine" that could drift.

## When generation isn't available

If torch/transformers aren't installed or the model isn't downloaded, the
Playground shows a clear error telling you to install dependencies; every other
tab still renders from the on-disk results. The dashboard is useful both as a
live profiler and as a static research browser.
