#!/usr/bin/env python3
"""benchmark_runner.py — Benchmark baseline vs optimized GGUF MoE engines.

Usage:
  python benchmark_runner.py --engine baseline
  python benchmark_runner.py --engine optimized
  python benchmark_runner.py --engine both
  python benchmark_runner.py --engine both --model medium
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from ireng import storage as st
from ireng.config import SMALL_MODEL_ID, MEDIUM_MODEL_ID
from ireng.engine import LlamaMoEEngine, EngineError
from ireng.prompts import SUITE
from baseline_engine import BaselineEngine
from optimized_engine import OptimizedEngine


def _run_suite(engine: LlamaMoEEngine, label: str) -> dict:
    """Benchmark all prompts; return aggregate stats."""
    results = []
    t_suite_start = time.perf_counter()

    for p in SUITE:
        print(f"    [{label}] {p.id} ...", end="", flush=True)
        try:
            m = engine.generate(p)
            results.append(m)
            print(f" {m.tokens_per_second} tok/s")
        except Exception as e:
            print(f" ERROR: {e}")

    suite_total_s = round(time.perf_counter() - t_suite_start, 2)

    if not results:
        return {"label": label, "error": "no results"}

    tps_vals = [r.tokens_per_second for r in results if r.tokens_per_second]
    mem_vals = [r.peak_memory_mb for r in results if r.peak_memory_mb]

    agg = {
        "label":             label,
        "mean_tps":          round(sum(tps_vals) / len(tps_vals), 4) if tps_vals else None,
        "peak_memory_mb":    round(max(mem_vals), 1) if mem_vals else None,
        "suite_total_s":     suite_total_s,
        "n_prompts":         len(results),
        "data_source":       "measured",
        "timestamp":         st.now_iso(),
    }

    # Write per-prompt rows to history
    rows = []
    for r in results:
        rows.append({
            "label": label, "prompt_id": r.prompt_id,
            "category": r.category,
            "tokens_generated": r.tokens_generated,
            "tokens_per_second": r.tokens_per_second,
            "time_to_first_token_s": r.time_to_first_token_s,
            "total_runtime_s": r.total_runtime_s,
            "peak_memory_mb": r.peak_memory_mb,
            "current_memory_mb": r.current_memory_mb,
            "cpu_utilization_pct": r.cpu_utilization_pct,
            "expert_bytes_per_tok_mb": r.expert_bytes_per_tok,
            "page_cache_hit_rate": r.page_cache_hit_rate,
            "context_length": r.context_length,
            "device": r.device,
            "measured": True,
            "data_source": "measured",
            "timestamp": st.now_iso(),
        })
    st.append_benchmark_history(rows)

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["baseline", "optimized", "both"],
                    default="both")
    ap.add_argument("--model", choices=["small", "medium"], default="small")
    args = ap.parse_args()

    model_id = MEDIUM_MODEL_ID if args.model == "medium" else SMALL_MODEL_ID
    print(f"Model: {model_id}")

    baseline_agg = optimized_agg = None

    if args.engine in ("baseline", "both"):
        print("\n── Baseline Engine ──")
        try:
            eng = BaselineEngine(model_id)
            eng.load()
            baseline_agg = _run_suite(eng, "baseline")
            eng.unload()
        except EngineError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if args.engine in ("optimized", "both"):
        print("\n── Optimized Engine ──")
        try:
            eng = OptimizedEngine(model_id)
            eng.load()
            optimized_agg = _run_suite(eng, "optimized")
            eng.unload()
        except EngineError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if baseline_agg and optimized_agg:
        b = baseline_agg["mean_tps"] or 1e-9
        o = optimized_agg["mean_tps"] or 0
        speedup = round(o / b, 3)
        comparison = {
            "baseline_tps":  baseline_agg["mean_tps"],
            "optimized_tps": optimized_agg["mean_tps"],
            "speedup":       speedup,
            "baseline_mem":  baseline_agg.get("peak_memory_mb"),
            "optimized_mem": optimized_agg.get("peak_memory_mb"),
            "data_source":   "measured",
            "model_id":      model_id,
            "timestamp":     st.now_iso(),
        }
        st.write_json(st.BEST_CONFIG_JSON.replace("best_config", "last_comparison"),
                      comparison)
        print(f"\nSpeedup: {speedup}×  "
              f"({baseline_agg['mean_tps']} → {optimized_agg['mean_tps']} tok/s)")

    elif baseline_agg:
        print(f"\nBaseline: {baseline_agg['mean_tps']} tok/s")
    elif optimized_agg:
        print(f"\nOptimized: {optimized_agg['mean_tps']} tok/s")


if __name__ == "__main__":
    main()
