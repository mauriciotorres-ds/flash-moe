# The Plots — How They Work

This document explains the static charts on their own. For the engine see
[ENGINE.md](ENGINE.md); for the live dashboard see [DASHBOARD.md](DASHBOARD.md).

The plot generator turns `results/experiments.csv` into a set of PNGs in
`plots/`. They are the "optimization timeline" view of the project — useful in
the README, the final report, slides, or anywhere you can't run Streamlit.

## Generate

```bash
pip install -r requirements.txt        # matplotlib
python plots/generate_plots.py
```

It reads whatever is on disk. With SAMPLE data loaded, every chart gets a
diagonal **"SAMPLE DATA"** watermark; with measured data the watermark is
absent. Either way the bottom-right corner is labelled
**"Hardware target: Apple M4 · 24 GB unified memory"** so a chart is never
ambiguous about the machine it represents.

## The six charts

**`01_tps_timeline.png` — Throughput by experiment.** tok/s for each experiment
in order. Points are coloured green (kept) or red (discarded); dashed line =
baseline, dotted line = best. This is the single most informative chart: you can
see the staircase of accepted wins and the failed attempts that didn't clear the
bar.

**`02_latency_timeline.png` — Mean latency by experiment.** Mean total latency
(seconds) per experiment. The mirror image of throughput; useful for spotting
optimizations that help tok/s but not single-request latency (e.g. batching).

**`03_memory_timeline.png` — Peak memory by experiment.** Peak process memory
(MB). Matters because of the 24 GB unified-memory ceiling on the M4 — half
precision and offloaded/static caches show up here.

**`04_speedup_timeline.png` — Speedup vs baseline.** Each experiment's tok/s
divided by the baseline, with a reference line at 1.0×. The headline
"how much faster did we get" curve.

**`05_keep_vs_discard.png` — Decision breakdown.** Bar chart of how many
experiments were kept, discarded, or marked n/a. Communicates the autoresearch
discipline: most ideas fail, and that's the point.

**`06_category_best.png` — Best throughput per category.** The best tok/s
achieved within each optimization category (runtime, memory, quantization,
scheduling, decoding, system, ssd_streaming). Shows which families of
optimization paid off most on this hardware.

## How they relate to the data

The generator is intentionally dumb: it does no math the runner didn't already
record. Colours come from the `decision` column, the baseline line from the
exp000 row, the watermark from the `data_source` column. So the plots are always
consistent with `experiments.csv`, the leaderboard, and the dashboard — there is
one source of truth.

## Refreshing after a real run

Once you have run the experiments for real on your Mac
(`python run_experiments.py --mode real --device mps`), just re-run
`python plots/generate_plots.py`. The watermark disappears, the numbers are
yours, and the hardware label still reads Apple M4 · 24 GB.
