#!/usr/bin/env python3
"""generate_plots.py — render the optimization-timeline charts.

Reads results/experiments.csv and produces PNGs in plots/. Every chart is
labelled with the target hardware (Apple M4 · 24 GB unified memory) and carries
a watermark when the underlying data is SAMPLE_SYNTHETIC, so a synthetic demo
chart can never be mistaken for a measured result.

Charts produced:
  01_tps_timeline.png        tok/s by experiment number (+ baseline / best line)
  02_latency_timeline.png    mean latency by experiment
  03_memory_timeline.png     peak memory by experiment
  04_speedup_timeline.png    speedup vs baseline by experiment
  05_keep_vs_discard.png     decision breakdown
  06_category_best.png       best tok/s achieved per optimization category

Usage:  python plots/generate_plots.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ireng import storage as st
from ireng.hardware import TARGET_SPEC

HW = TARGET_SPEC["label"]
OUT = st.PLOTS


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load():
    rows = st.read_experiments()
    rows = [r for r in rows if r.get("exp") not in (None, "")]
    rows.sort(key=lambda r: int(float(r["exp"])))
    return rows


def _is_sample(rows):
    return any(r.get("data_source") == "SAMPLE_SYNTHETIC" for r in rows)


def _decorate(fig, ax, title, sample):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.text(0.99, 0.01, f"Hardware target: {HW}", ha="right", va="bottom",
             fontsize=8, color="#555")
    if sample:
        fig.text(0.5, 0.5, "SAMPLE DATA", fontsize=46, color="red",
                 alpha=0.12, ha="center", va="center", rotation=25,
                 fontweight="bold")


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def main():
    rows = _load()
    if not rows:
        print("No experiment data. Run seed_sample_data.py or the runner first.")
        return 1
    sample = _is_sample(rows)
    xs = [int(float(r["exp"])) for r in rows]
    tps = [_f(r.get("mean_tps")) for r in rows]
    lat = [_f(r.get("mean_latency_s")) for r in rows]
    mem = [_f(r.get("peak_memory_mb")) for r in rows]
    decisions = [r.get("decision", "") for r in rows]
    baseline = next((_f(r.get("mean_tps")) for r in rows if int(float(r["exp"])) == 0), None)

    # 01 tok/s timeline
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"keep": "#2a9d8f", "discard": "#e76f51", "n/a": "#888", "": "#888"}
    ax.plot(xs, [t if t else None for t in tps], "-", color="#264653", alpha=0.5, zorder=1)
    for x, t, d in zip(xs, tps, decisions):
        if t:
            ax.scatter([x], [t], color=colors.get(d, "#888"), zorder=2, s=30)
    if baseline:
        ax.axhline(baseline, ls="--", color="#999", label=f"baseline ({baseline:.1f})")
    best = max([t for t in tps if t] or [0])
    ax.axhline(best, ls=":", color="#2a9d8f", label=f"best ({best:.1f})")
    ax.set_xlabel("experiment #"); ax.set_ylabel("tok/s")
    ax.legend(loc="best", fontsize=8)
    _decorate(fig, ax, "Throughput by experiment (green=keep, red=discard)", sample)
    _save(fig, "01_tps_timeline.png")

    # 02 latency timeline
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs, lat, "o-", color="#e9c46a")
    ax.set_xlabel("experiment #"); ax.set_ylabel("mean latency (s)")
    _decorate(fig, ax, "Mean latency by experiment", sample)
    _save(fig, "02_latency_timeline.png")

    # 03 memory timeline
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs, mem, "s-", color="#8d99ae")
    ax.set_xlabel("experiment #"); ax.set_ylabel("peak memory (MB)")
    _decorate(fig, ax, "Peak memory by experiment", sample)
    _save(fig, "03_memory_timeline.png")

    # 04 speedup timeline
    fig, ax = plt.subplots(figsize=(10, 4.5))
    spd = [(t / baseline) if (t and baseline) else None for t in tps]
    ax.plot(xs, spd, "d-", color="#457b9d")
    ax.axhline(1.0, ls="--", color="#999")
    ax.set_xlabel("experiment #"); ax.set_ylabel("speedup vs baseline (x)")
    _decorate(fig, ax, "Speedup vs baseline by experiment", sample)
    _save(fig, "04_speedup_timeline.png")

    # 05 keep vs discard
    fig, ax = plt.subplots(figsize=(6, 4.5))
    from collections import Counter
    c = Counter(decisions)
    labels = list(c.keys()); vals = [c[k] for k in labels]
    ax.bar(labels, vals, color=[colors.get(k, "#888") for k in labels])
    ax.set_ylabel("# experiments")
    _decorate(fig, ax, "Decisions: keep vs discard vs n/a", sample)
    _save(fig, "05_keep_vs_discard.png")

    # 06 best tok/s per category
    fig, ax = plt.subplots(figsize=(9, 4.5))
    from collections import defaultdict
    bycat = defaultdict(float)
    for r in rows:
        t = _f(r.get("mean_tps"))
        if t:
            bycat[r.get("category", "?")] = max(bycat[r.get("category", "?")], t)
    cats = list(bycat.keys()); vals = [bycat[k] for k in cats]
    ax.bar(cats, vals, color="#2a9d8f")
    ax.set_ylabel("best tok/s in category")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _decorate(fig, ax, "Best throughput achieved per optimization category", sample)
    _save(fig, "06_category_best.png")

    print(f"\nDone. {'(SAMPLE watermark applied)' if sample else '(measured data)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
