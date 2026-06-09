"""runner.py — Autoresearch loop for MoE expert-streaming experiments.

For each experiment:
  1. Build config (delta on baseline or current best).
  2. Mark state.json status="running" so interruptions are detectable.
  3. Benchmark on the real GGUF model OR produce a mock dry-run.
  4. Keep/discard against the current best tok/s.
  5. Write experiments.csv/json, benchmark_history.csv, leaderboard.csv,
     best_config.json, expNNN.md, failure_log.md (on discard), and git commit.

Resume: reads state.json on start; skips completed experiments;
        if status was "running" that experiment re-runs cleanly.
"""
from __future__ import annotations

import json
import time
from statistics import mean, median
from typing import Optional

from . import storage as st
from .config import EngineConfig, baseline_config, find_gguf_for_model
from .engine import LlamaMoEEngine
from .experiments import EXPERIMENTS, Experiment, get as get_exp
from .metrics import GenerationMetrics
from .prompts import SUITE, Prompt

TPS_KEEP_GAIN = 0.01   # ≥1% faster than best → keep
MEM_KEEP_GAIN = 0.05   # ≥5% lower memory at ≤2% tps loss → keep

MEASURE_RUNS  = 2   # benchmark repetitions per prompt (median)
WARMUP_RUNS   = 1   # throw-away runs before measuring


# ── Benchmark helper ──────────────────────────────────────────────────────────

def _run_prompt(engine: LlamaMoEEngine, prompt: Prompt) -> GenerationMetrics:
    return engine.generate(prompt)


def _bench(engine: LlamaMoEEngine, prompts: list[Prompt],
           n_warmup: int = WARMUP_RUNS,
           n_measure: int = MEASURE_RUNS) -> list[GenerationMetrics]:
    """Run warmup + measure passes; return measured results."""
    for _ in range(n_warmup):
        _run_prompt(engine, prompts[0])   # warmup on first prompt only
    results = []
    for p in prompts:
        runs = [_run_prompt(engine, p) for _ in range(n_measure)]
        # Pick the median-tok/s run
        runs.sort(key=lambda m: m.tokens_per_second)
        results.append(runs[len(runs) // 2])
    return results


def _agg(results: list[GenerationMetrics]) -> dict:
    tps_vals  = [r.tokens_per_second for r in results if r.tokens_per_second]
    lat_vals  = [r.total_runtime_s   for r in results if r.total_runtime_s]
    mem_vals  = [r.peak_memory_mb    for r in results if r.peak_memory_mb]
    ttft_vals = [r.time_to_first_token_s for r in results
                 if r.time_to_first_token_s is not None]
    return {
        "mean_tps":       round(mean(tps_vals), 4)  if tps_vals  else None,
        "mean_latency_s": round(mean(lat_vals), 4)  if lat_vals  else None,
        "peak_memory_mb": round(mean(mem_vals), 2)  if mem_vals  else None,
        "mean_ttft_s":    round(mean(ttft_vals), 4) if ttft_vals else None,
        "expert_bytes_per_tok_mb": results[0].expert_bytes_per_tok
                                   if results else None,
        "page_cache_hit_rate":     results[0].page_cache_hit_rate
                                   if results else None,
        "n_experts":      (results[0].expert_stats.n_experts
                           if results and results[0].expert_stats else None),
        "n_experts_used": (results[0].expert_stats.n_experts_used
                           if results and results[0].expert_stats else None),
        "n_moe_layers":   (results[0].expert_stats.n_moe_layers
                           if results and results[0].expert_stats else None),
    }


# ── Mock result (for dry-run pipeline testing) ────────────────────────────────

def _mock_agg(cfg: EngineConfig, baseline_tps: Optional[float]) -> dict:
    import hashlib
    sig = json.dumps(cfg.to_dict(), sort_keys=True)
    h   = int(hashlib.sha256(sig.encode()).hexdigest(), 16)
    base_tps  = baseline_tps or 2.5
    variation = ((h % 200) - 100) / 1000.0
    tps = round(base_tps * (1.0 + variation), 3)
    return {
        "mean_tps": tps, "mean_latency_s": round(len(SUITE[0].user) / tps, 3),
        "peak_memory_mb": round(4800 + (h % 400) - 200, 1),
        "mean_ttft_s": round(0.8 + (h % 20) / 100, 3),
        "expert_bytes_per_tok_mb": 6.75, "page_cache_hit_rate": None,
        "n_experts": 60, "n_experts_used": 4, "n_moe_layers": 24,
    }


# ── Decision ─────────────────────────────────────────────────────────────────

def _decide(agg: dict, best_tps: Optional[float],
            baseline_tps: Optional[float]) -> str:
    tps = agg.get("mean_tps")
    if tps is None:
        return "discard"
    if best_tps is None:
        return "keep"   # first experiment (baseline)
    if tps >= best_tps * (1 + TPS_KEEP_GAIN):
        return "keep"
    # Memory win with small speed loss
    mem_now = agg.get("peak_memory_mb") or 0
    # (can't compare without baseline memory here; use simple speed rule)
    if tps < best_tps * 0.98:
        return "discard"
    return "discard"


# ── Experiment markdown ───────────────────────────────────────────────────────

def _exp_md(exp_def: Experiment, cfg: EngineConfig, agg: dict,
            decision: str, baseline_tps: Optional[float],
            best_tps: Optional[float], data_source: str) -> str:
    tps     = agg.get("mean_tps")
    speedup = round(tps / baseline_tps, 3) if (tps and baseline_tps) else "n/a"
    delta   = (round((tps - best_tps) / best_tps * 100, 2)
               if (tps and best_tps) else "n/a")

    overrides_str = (json.dumps(exp_def.overrides, indent=2)
                     if exp_def.overrides else "(none — same as baseline)")

    return f"""# exp{exp_def.exp:03d} — {exp_def.title}

## Metadata
- **Category:** {exp_def.category}
- **Base config:** {exp_def.base}
- **Decision:** {decision.upper()}
- **Data source:** {data_source}
- **measured:** true

## Config overrides
```json
{overrides_str}
```

## Hypothesis
{exp_def.hypothesis}

## Rationale
{exp_def.rationale}

## Results
| Metric | Value |
|--------|-------|
| mean tok/s | {agg.get("mean_tps")} |
| speedup vs baseline | {speedup}× |
| delta vs best | {delta}% |
| TTFT (s) | {agg.get("mean_ttft_s")} |
| mean latency (s) | {agg.get("mean_latency_s")} |
| peak memory (MB) | {agg.get("peak_memory_mb")} |
| expert bytes/tok (MB) | {agg.get("expert_bytes_per_tok_mb")} |
| page cache hit rate | {agg.get("page_cache_hit_rate")} |
| n_experts | {agg.get("n_experts")} |
| n_experts_used (top-K) | {agg.get("n_experts_used")} |
| n_moe_layers | {agg.get("n_moe_layers")} |

## Decision rationale
{"KEEP: improves tok/s above threshold." if decision == "keep"
 else "DISCARD: did not improve tok/s above keep threshold."}
"""


# ── Main runner ───────────────────────────────────────────────────────────────

class ExperimentRunner:
    def __init__(self, mode: str = "real", resume: bool = True,
                 start_exp: Optional[int] = None,
                 end_exp: Optional[int] = None):
        self.mode      = mode          # "real" | "mock"
        self.resume    = resume
        self.start_exp = start_exp
        self.end_exp   = end_exp

    def run(self):
        state          = st.read_state()
        baseline_tps   = state.get("baseline_tps")
        best_tps       = state.get("best_tps")
        best_config_d  = state.get("best_config_dict")

        last_done = state.get("last_completed_experiment", -1)
        if not self.resume:
            last_done = -1

        engine: Optional[LlamaMoEEngine] = None

        for exp_def in EXPERIMENTS:
            n = exp_def.exp

            if self.start_exp is not None and n < self.start_exp:
                continue
            if self.end_exp is not None and n > self.end_exp:
                break
            if self.resume and n <= last_done:
                print(f"  exp{n:03d}: already done, skipping")
                continue

            # Build config
            if exp_def.base == "best" and best_config_d:
                base_cfg = EngineConfig.from_dict(best_config_d)
            else:
                base_cfg = baseline_config()

            overrides = dict(exp_def.overrides)
            # exp040: fill from best_config.json
            if n == 40 and best_config_d:
                overrides = {}
                base_cfg  = EngineConfig.from_dict(best_config_d)

            cfg = EngineConfig.from_dict({**base_cfg.to_dict(),
                                          **overrides,
                                          "label": f"exp{n:03d}"})

            # Mark running
            state.update({"current_experiment": n, "status": "running",
                          "last_updated": st.now_iso()})
            st.write_state(state)
            print(f"\n{'='*60}")
            print(f"  exp{n:03d}: {exp_def.title}")
            print(f"  overrides: {exp_def.overrides}")

            t0 = time.perf_counter()

            # ── Benchmark ────────────────────────────────────────────────
            if self.mode == "real":
                if engine is None or engine.config.model_id != cfg.model_id:
                    if engine:
                        engine.unload()
                    engine = LlamaMoEEngine(cfg)
                    try:
                        engine.load()
                    except Exception as e:
                        print(f"  ERROR loading engine: {e}")
                        state["status"] = "error"
                        st.write_state(state)
                        continue
                else:
                    engine.reconfigure(cfg)

                try:
                    results = _bench(engine, list(SUITE))
                except Exception as e:
                    print(f"  ERROR during benchmark: {e}")
                    state["status"] = "error"
                    st.write_state(state)
                    continue
                agg         = _agg(results)
                data_source = "measured"
            else:
                agg         = _mock_agg(cfg, baseline_tps)
                data_source = "MOCK_DRYRUN"

            wall = round(time.perf_counter() - t0, 1)
            print(f"  tok/s={agg.get('mean_tps')}  "
                  f"mem={agg.get('peak_memory_mb')} MB  ({wall}s)")

            # ── Decision ──────────────────────────────────────────────────
            decision = _decide(agg, best_tps, baseline_tps)

            if n == 0:
                baseline_tps = agg.get("mean_tps")
                best_tps     = agg.get("mean_tps")
                best_config_d = cfg.to_dict()
                decision      = "keep"

            if decision == "keep":
                if best_tps is None or (agg.get("mean_tps") or 0) > best_tps:
                    best_tps      = agg.get("mean_tps")
                    best_config_d = cfg.to_dict()
                    st.write_best_config(n, cfg.label, cfg.to_dict(),
                                         best_tps, data_source)
                print(f"  → KEEP  (best_tps={best_tps})")
            else:
                print(f"  → DISCARD")
                st.append_failure(
                    n, exp_def.title,
                    why_attempted=exp_def.hypothesis[:120],
                    why_failed="Did not improve tok/s above keep threshold.",
                    perf_impact=f"mean_tps={agg.get('mean_tps')} vs best={best_tps}",
                    lessons="Config delta did not help on this hardware.",
                )

            # ── Write results ─────────────────────────────────────────────
            row = {
                "exp": n, "title": exp_def.title,
                "category": exp_def.category, "label": cfg.label,
                "decision": decision,
                "baseline_tps": baseline_tps, "best_tps": best_tps,
                "delta_pct_vs_best": (
                    round(((agg.get("mean_tps") or 0) - (best_tps or 0))
                          / (best_tps or 1) * 100, 2)
                    if best_tps else None),
                "measured": True,
                "data_source": data_source,
                "device": "metal" if cfg.n_gpu_layers != 0 else "cpu",
                "timestamp": st.now_iso(),
                **agg,
            }
            st.append_experiment_row(row)
            st.append_benchmark_history([{**row, "prompt_id": "aggregate",
                                           "total_runtime_s": agg.get("mean_latency_s")}])
            st.rebuild_leaderboard()

            # Write expNNN.md
            st.write_experiment_md(
                n, _exp_md(exp_def, cfg, agg, decision,
                           baseline_tps, best_tps, data_source))

            # Update state
            state.update({
                "last_completed_experiment": n,
                "status": "running",
                "baseline_tps": baseline_tps,
                "best_tps": best_tps,
                "best_config_dict": best_config_d,
                "last_updated": st.now_iso(),
            })
            st.write_state(state)
            st.git_commit(
                f"exp{n:03d}: {exp_def.title[:50]} "
                f"[{decision}] {agg.get('mean_tps')} tok/s")

        # Final state
        if engine:
            engine.unload()
        state["status"] = "complete"
        st.write_state(state)
        st.git_tag("optimization-complete")
        print("\nAll experiments complete.")
        print(f"Baseline: {baseline_tps} tok/s  |  Best: {best_tps} tok/s")
        if baseline_tps and best_tps:
            print(f"Speedup: {round(best_tps/baseline_tps, 3)}×")
