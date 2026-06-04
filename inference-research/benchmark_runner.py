#!/usr/bin/env python3
"""benchmark_runner.py — run the reproducible benchmark suite.

Benchmarks the baseline engine, the optimized engine, or both, across all five
prompt categories (factual, coding, reasoning, summarization, structured), and
writes per-prompt rows to results/benchmark_history.csv plus a comparison
summary. Computes speedup = optimized_tps / baseline_tps.

Usage:
  python benchmark_runner.py --engine baseline
  python benchmark_runner.py --engine optimized
  python benchmark_runner.py --engine both            # side-by-side + speedup
  python benchmark_runner.py --engine both --device mps
"""
from __future__ import annotations

import argparse
import json

from ireng.config import baseline_config
from ireng.benchmark import run_benchmark, AggregateResult
from ireng.prompts import SUITE, CATEGORIES
from ireng.hardware import TARGET_SPEC, detect_host
from ireng import storage as st
from optimized_engine import load_optimized_config


def _history_rows(tag, agg: AggregateResult):
    rows = []
    for pp in agg.per_prompt:
        rows.append({"exp": tag, "label": agg.label, **pp,
                     "measured": True, "data_source": "MEASURED",
                     "timestamp": st.now_iso()})
    return rows


def _print_agg(name, agg: AggregateResult):
    print(f"\n=== {name} ({agg.device}) ===")
    print(f"  mean tok/s     : {agg.mean_tps}")
    print(f"  decode tok/s   : {agg.mean_decode_tps}")
    print(f"  mean TTFT (s)  : {agg.mean_ttft_s}")
    print(f"  mean latency   : {agg.mean_latency_s}")
    print(f"  peak mem (MB)  : {agg.peak_memory_mb}")
    print(f"  GPU util       : {agg.mean_gpu_pct if agg.mean_gpu_pct is not None else 'n/a (MPS)'}")
    print(f"  quality score  : {agg.quality_score}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["baseline", "optimized", "both"], default="both")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--category", choices=CATEGORIES, default=None,
                    help="restrict to one prompt category")
    a = ap.parse_args()

    print(f"Target: {TARGET_SPEC['label']}  |  host device: {detect_host().device}")
    prompts = [p for p in SUITE if (a.category is None or p.category == a.category)]

    results = {}
    if a.engine in ("baseline", "both"):
        cfg = baseline_config(a.model).delta(device=a.device)
        results["baseline"] = run_benchmark(cfg, prompts)
        st.append_benchmark_history(_history_rows("baseline", results["baseline"]))
        _print_agg("BASELINE", results["baseline"])

    if a.engine in ("optimized", "both"):
        cfg = load_optimized_config(a.model, a.device)
        results["optimized"] = run_benchmark(cfg, prompts)
        st.append_benchmark_history(_history_rows("optimized", results["optimized"]))
        _print_agg("OPTIMIZED", results["optimized"])

    if a.engine == "both":
        b, o = results["baseline"].mean_tps, results["optimized"].mean_tps
        speedup = round(o / b, 3) if b else None
        print(f"\n>>> speedup = optimized/baseline = {speedup}x "
              f"({o} / {b} tok/s)")
        st.write_json(st.RESULTS + "/last_comparison.json", {
            "baseline_tps": b, "optimized_tps": o, "speedup": speedup,
            "device": results["baseline"].device, "timestamp": st.now_iso(),
            "data_source": "MEASURED",
        })


if __name__ == "__main__":
    main()
