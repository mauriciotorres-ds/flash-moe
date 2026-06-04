"""Autoresearch runner — the loop that turns the registry into results on disk.

For each experiment it:
  1. builds the config (delta on current-best, or on baseline for ablations),
  2. writes state.json status="running" (so an interruption is detectable),
  3. benchmarks it (REAL on the host, or MOCK for pipeline verification),
  4. decides keep / discard against the current best using measured numbers,
  5. writes experiments.csv/json, benchmark_history.csv, leaderboard.csv,
     best_config.json, the expNNN.md write-up, and failure_log.md on discard,
  6. commits to git.

Resume protocol: on start it reads state.json and skips completed experiments.
If the previous status was "running", that experiment is re-run cleanly.

Modes
-----
--real : run the real model on the host (Apple M4 target). Requires torch +
         transformers + the model downloaded.
--mock : deterministic synthetic numbers to verify the pipeline end-to-end
         WITHOUT a model. Every mock row is flagged measured=false,
         data_source="MOCK_DRYRUN" and must never be read as a real result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Dict, Optional

from . import storage as st
from .config import EngineConfig, baseline_config
from .experiments import REGISTRY, Experiment, get as get_exp
from .benchmark import AggregateResult, run_benchmark

# Decision thresholds
TPS_KEEP_GAIN = 0.01      # >=1% faster than best -> keep on speed
MEM_KEEP_GAIN = 0.05      # >=5% lower peak memory at <=2% tps loss -> keep
QUALITY_FLOOR = 0.7


# --------------------------------------------------------------------- mock
def _mock_result(cfg: EngineConfig, baseline: Optional[AggregateResult]) -> AggregateResult:
    """Deterministic synthetic aggregate from the config signature.

    Purpose: exercise every code path (decision, storage, plots, dashboard)
    without a model. Clearly synthetic; never a real measurement.
    """
    sig = json.dumps(cfg.signature(), sort_keys=True)
    h = int(hashlib.sha256(sig.encode()).hexdigest(), 16)
    base_tps = 14.0
    # plausible deltas keyed off knobs (illustrative only)
    mult = 1.0
    s = cfg.signature()
    if s["dtype"] in ("float16", "bfloat16"):
        mult *= 1.6
    if s["attn_implementation"] == "sdpa":
        mult *= 1.12
    if s["attn_implementation"] == "flash_attention_2":
        mult *= 0.0 if cfg.device == "mps" else 1.1  # unsupported on mps
    if s["torch_compile"]:
        mult *= 1.18
    if s["inference_mode"]:
        mult *= 1.03
    if s["no_grad"]:
        mult *= 1.02
    if not s["use_cache"]:
        mult *= 0.35
    if not s["eval_mode"]:
        mult *= 0.9
    if s["cache_implementation"] == "static":
        mult *= 1.04
    if s["batch_size"] and s["batch_size"] > 1:
        mult *= 1.0 + 0.15 * (s["batch_size"] ** 0.5)
    if s["quantization"] != "none" and cfg.device == "mps":
        mult *= 0.0  # bnb unsupported -> treated as no-run/fallback
    jitter = 1.0 + ((h % 1000) / 1000.0 - 0.5) * 0.04  # +-2%
    tps = round(base_tps * mult * jitter, 3) if mult > 0 else 0.0

    quality = 1.0
    if not s["eval_mode"]:
        quality = 0.5
    if cfg.quantization in ("nf4", "fp4"):
        quality = 0.6
    if cfg.max_tokens_scale < 0.8:
        quality = 0.92

    peak = 1100.0 / (1.0 if s["dtype"] == "float32" else 1.8)
    return AggregateResult(
        label=cfg.label, mean_tps=tps, median_tps=tps,
        mean_decode_tps=round(tps * 1.1, 3),
        mean_ttft_s=round(0.18 / max(mult, 0.2), 4) if mult else None,
        mean_latency_s=round(8.0 / max(tps, 0.1), 4),
        peak_memory_mb=round(peak, 1),
        mean_cpu_pct=round(60 + (h % 30), 1),
        mean_gpu_pct=None,  # MPS: not available
        kv_cache_mb=round(2.0 + (h % 5), 3),
        quality_ok=quality >= QUALITY_FLOOR, quality_score=quality,
        device=cfg.device, per_prompt=[],
        support_notes=["MOCK_DRYRUN — synthetic numbers, not a measurement."],
    )


# ---------------------------------------------------------------- decision
def decide(result: AggregateResult, best: AggregateResult,
           exp: Experiment) -> str:
    if result.quality_score < QUALITY_FLOOR:
        return "discard"
    if best is None or best.mean_tps <= 0:
        return "keep"
    if result.mean_tps <= 0:
        return "discard"
    gain = (result.mean_tps - best.mean_tps) / best.mean_tps
    if gain >= TPS_KEEP_GAIN:
        return "keep"
    # memory-only win path
    if (best.peak_memory_mb and result.peak_memory_mb
            and result.peak_memory_mb <= best.peak_memory_mb * (1 - MEM_KEEP_GAIN)
            and gain >= -0.02):
        return "keep"
    return "discard"


# ------------------------------------------------------- experiment write-up
def _exp_md(exp: Experiment, cfg: EngineConfig, result: AggregateResult,
            baseline: AggregateResult, best: AggregateResult, decision: str,
            data_source: str) -> str:
    delta_best = ("n/a" if not best or best.mean_tps <= 0 else
                  f"{(result.mean_tps - best.mean_tps) / best.mean_tps * 100:+.2f}%")
    delta_base = ("n/a" if not baseline or baseline.mean_tps <= 0 else
                  f"{(result.mean_tps - baseline.mean_tps) / baseline.mean_tps * 100:+.2f}%")
    notes = "\n".join(f"  - {n}" for n in result.support_notes) or "  - none"
    return f"""# Experiment {exp.exp:03d} — {exp.title}

- **Category:** {exp.category}
- **Applied on:** {exp.base} config
- **Data source:** {data_source} (`measured={str(data_source == 'MEASURED').lower()}`)
- **Decision:** **{decision.upper()}**

## Hypothesis
{exp.hypothesis}

## Rationale (bottleneck targeted)
{exp.rationale}

## Prior expectation
{exp.prior}

## Implementation
Config delta applied to the {exp.base} configuration:

```json
{json.dumps(exp.overrides, indent=2)}
```

Resulting engine signature:

```json
{json.dumps(cfg.signature(), indent=2)}
```

## Files Modified
- `ireng/experiments.py` (registry entry)
- engine behaviour driven by `ireng/engine.py` (config-driven; no code fork)

## Benchmark Results
| metric | value |
|---|---|
| mean tok/s | {result.mean_tps} |
| decode tok/s | {result.mean_decode_tps} |
| mean TTFT (s) | {result.mean_ttft_s} |
| mean latency (s) | {result.mean_latency_s} |
| peak memory (MB) | {result.peak_memory_mb} |
| CPU util (%) | {result.mean_cpu_pct} |
| GPU util (%) | {result.mean_gpu_pct if result.mean_gpu_pct is not None else "n/a (MPS)"} |
| KV cache (MB) | {result.kv_cache_mb} |
| quality score | {result.quality_score} |
| device | {result.device} |

## Performance Change
- vs **baseline**: {delta_base}
- vs **current best** ({best.label if best else "—"}): {delta_best}

## Support Notes
{notes}

## Decision
**{decision.upper()}** — kept only if it improved throughput/latency/memory
without degrading output quality (quality floor {QUALITY_FLOOR}).

## Lessons Learned
{exp.prior}
"""


# ------------------------------------------------------------------- driver
_DATA_SOURCE = {"real": "MEASURED", "mock": "MOCK_DRYRUN", "sample": "SAMPLE_SYNTHETIC"}


def run(mode: str = "mock", only=None, resume: bool = True,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto",
        commit: bool = True):
    data_source = _DATA_SOURCE.get(mode, "MOCK_DRYRUN")
    measured = mode == "real"

    state = st.read_state()
    base_cfg = baseline_config(model_id).delta(device=device)

    # cache of result objects this session for leaderboard/best tracking
    best_cfg = base_cfg
    best_result: Optional[AggregateResult] = None
    baseline_result: Optional[AggregateResult] = None

    def _bench(cfg: EngineConfig) -> AggregateResult:
        if mode == "real":
            return run_benchmark(cfg)
        r = _mock_result(cfg, baseline_result)
        r.support_notes = [f"{data_source} — synthetic numbers, not a measurement."]
        return r

    # ---- exp000: baseline -------------------------------------------------
    run_baseline = only is None or 0 in only
    if run_baseline and (not resume or state.get("last_completed_experiment", -1) < 0):
        state.update(phase=1, current_experiment=0, status="running")
        st.write_state(state)
        baseline_result = _bench(base_cfg)
        best_result = baseline_result
        _persist(0, _baseline_exp(), base_cfg, baseline_result, baseline_result,
                 baseline_result, "keep", data_source, measured, commit)
        state.update(last_completed_experiment=0, baseline_tps=baseline_result.mean_tps,
                     best_tps=best_result.mean_tps, best_config="exp000", status="idle")
        st.write_state(state)
    else:
        # resume: reconstruct baseline/best numbers from disk
        bc = st.read_json(st.BEST_CONFIG_JSON, {})
        state_best = state.get("best_tps")
        baseline_result = AggregateResult(label="baseline",
                                          mean_tps=state.get("baseline_tps") or 0.0)
        best_result = AggregateResult(label=bc.get("label", "baseline"),
                                      mean_tps=state_best or baseline_result.mean_tps)

    # ---- experiments ------------------------------------------------------
    for exp in REGISTRY:
        if only is not None and exp.exp not in only:
            continue
        if resume and exp.exp <= state.get("last_completed_experiment", -1):
            continue

        # special-case: MoE SSD streaming is N/A on the dense dev model
        if exp.exp == 32:
            _persist_na(exp, base_cfg, data_source, commit)
            state.update(last_completed_experiment=exp.exp, current_experiment=exp.exp + 1,
                         status="idle")
            st.write_state(state)
            continue

        parent = best_cfg if exp.base == "best" else base_cfg
        cfg = parent.delta(**exp.overrides)

        state.update(current_experiment=exp.exp, status="running")
        st.write_state(state)

        result = _bench(cfg)
        decision = decide(result, best_result, exp)

        _persist(exp.exp, exp, cfg, result, baseline_result, best_result,
                 decision, data_source, measured, commit)

        if decision == "keep":
            best_cfg = cfg.delta(label=f"best@exp{exp.exp:03d}")
            best_result = result
            st.write_best_config(exp.exp, exp.title, best_cfg.as_dict(),
                                 result.mean_tps, data_source)
            state.update(best_tps=result.mean_tps, best_config=f"exp{exp.exp:03d}")
        else:
            _log_failure(exp, result, best_result)

        st.rebuild_leaderboard()
        state.update(last_completed_experiment=exp.exp,
                     current_experiment=exp.exp + 1, status="idle")
        st.write_state(state)

    if commit:
        st.git_tag("optimization-complete")
    print(f"[runner] done. mode={mode} best={state.get('best_config')} "
          f"best_tps={state.get('best_tps')}")
    return state


def _persist(exp_num, exp, cfg, result, baseline, best, decision,
             data_source, measured, commit):
    row = {
        "exp": exp_num, "title": exp.title, "category": exp.category,
        "label": cfg.label, "decision": decision,
        "mean_tps": result.mean_tps,
        "baseline_tps": baseline.mean_tps if baseline else None,
        "best_tps": best.mean_tps if best else None,
        "delta_pct_vs_best": (round((result.mean_tps - best.mean_tps) / best.mean_tps * 100, 2)
                              if best and best.mean_tps else None),
        "mean_ttft_s": result.mean_ttft_s, "mean_latency_s": result.mean_latency_s,
        "peak_memory_mb": result.peak_memory_mb, "kv_cache_mb": result.kv_cache_mb,
        "quality_ok": result.quality_ok, "measured": measured,
        "data_source": data_source, "device": result.device,
        "timestamp": st.now_iso(),
        "notes": "; ".join(result.support_notes)[:300],
    }
    st.append_experiment_row(row)
    st.append_benchmark_history([{
        "exp": exp_num, "label": cfg.label, "prompt_id": pp.get("prompt_id", ""),
        "category": pp.get("category", ""), **pp,
        "measured": measured, "data_source": data_source, "timestamp": st.now_iso(),
    } for pp in (result.per_prompt or [{}])])
    md = _exp_md(exp, cfg, result, baseline, best, decision, data_source)
    st.write_experiment_md(exp_num, md)
    if commit:
        st.git_commit(f"exp{exp_num:03d}: {exp.title} {result.mean_tps} tok/s — {decision}")


def _persist_na(exp, cfg, data_source, commit):
    row = {
        "exp": exp.exp, "title": exp.title, "category": exp.category,
        "label": exp.overrides.get("label", "na"), "decision": "n/a",
        "mean_tps": None, "quality_ok": True, "measured": False,
        "data_source": data_source, "device": cfg.device, "timestamp": st.now_iso(),
        "notes": "Dense dev model has no experts; validated in Phase 2 on 397B MoE.",
    }
    st.append_experiment_row(row)
    md = f"""# Experiment {exp.exp:03d} — {exp.title}

- **Category:** {exp.category}
- **Decision:** **N/A (dense model)**

## Hypothesis
{exp.hypothesis}

## Why N/A here
{exp.rationale}

This optimization only exists for Mixture-of-Experts models. The Phase 1 dev
model (Qwen2.5-0.5B-Instruct) is dense, so there are no per-token experts to
stream from SSD. The technique is therefore validated in **Phase 2** on
Qwen3.5-397B-A17B (see `large_model/validate_transfer.py`), reusing the
Flash-MoE Metal engine's pread-based expert streaming.

## Lessons Learned
{exp.prior}
"""
    st.write_experiment_md(exp.exp, md)
    if commit:
        st.git_commit(f"exp{exp.exp:03d}: {exp.title} — N/A on dense model (Phase 2)")


def _log_failure(exp, result, best):
    impact = ("quality below floor" if result.quality_score < QUALITY_FLOOR
              else f"{(result.mean_tps - best.mean_tps) / best.mean_tps * 100:+.2f}% vs best"
              if best and best.mean_tps else "no improvement")
    st.append_failure(
        exp.exp, exp.title,
        why_attempted=exp.hypothesis,
        why_failed=(f"Did not beat current best by >= {TPS_KEEP_GAIN*100:.0f}% "
                    f"or degraded quality. {'; '.join(result.support_notes)}"),
        perf_impact=impact,
        lessons=exp.prior,
    )


def _baseline_exp() -> Experiment:
    return Experiment(0, "Reproducible HF baseline", "runtime", "baseline", {},
                      "Plain Transformers fp32 eager is the reference point.",
                      "Establish the number every experiment is measured against.",
                      prior="This is the control.")


def main():
    ap = argparse.ArgumentParser(description="Autoresearch experiment runner")
    ap.add_argument("--mode", choices=["real", "mock", "sample"], default="mock",
                    help="real = run the model on this host; mock = synthetic dry-run; "
                         "sample = seed clearly-labelled demo data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="run only these experiment numbers")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args()
    run(mode=a.mode, only=a.only, resume=not a.no_resume, model_id=a.model,
        device=a.device, commit=not a.no_commit)


if __name__ == "__main__":
    main()
