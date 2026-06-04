"""Benchmark engine — runs the prompt suite against a config and aggregates.

Reproducible: fixed prompts, fixed seed, warmup runs discarded, median of N
measured runs per prompt. Returns per-prompt metrics plus an aggregate the
runner stores. Also performs a lightweight *quality* check so we can honour the
retention rule "preserve output quality" (a speedup that corrupts output is a
discard, mirroring Flash-MoE's 2-bit JSON failure).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .config import EngineConfig
from .engine import ConfigurableEngine
from .metrics import GenerationMetrics
from .prompts import SUITE, Prompt


@dataclass
class AggregateResult:
    label: str = ""
    mean_tps: float = 0.0
    median_tps: float = 0.0
    mean_decode_tps: float = 0.0
    mean_ttft_s: Optional[float] = None
    mean_latency_s: float = 0.0
    peak_memory_mb: Optional[float] = None
    mean_cpu_pct: Optional[float] = None
    mean_gpu_pct: Optional[float] = None
    kv_cache_mb: Optional[float] = None
    quality_ok: bool = True
    quality_score: float = 1.0
    device: str = ""
    per_prompt: List[Dict] = field(default_factory=list)
    support_notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


# ---- quality heuristics (no ground-truth model needed) --------------------
def _quality_score(prompt: Prompt, text: str) -> float:
    """Cheap, deterministic sanity checks. Returns 0..1.

    The point isn't accuracy grading; it's catching *corruption* introduced by
    an optimization (empty output, garbage, broken JSON for structured tasks).
    """
    t = (text or "").strip()
    if len(t) < 2:
        return 0.0
    score = 1.0
    # crude gibberish guard: ratio of non-printable / replacement chars
    bad = sum(1 for c in t if ord(c) < 9 or c == "�")
    if bad / max(len(t), 1) > 0.05:
        score -= 0.5
    if prompt.category == "structured":
        if prompt.id == "struct_json":
            import json
            # find first {...} block and try to parse
            s, e = t.find("{"), t.rfind("}")
            if s == -1 or e == -1:
                score -= 0.6
            else:
                try:
                    json.loads(t[s:e + 1])
                except Exception:
                    score -= 0.5
        elif prompt.id == "struct_table":
            if "|" not in t:
                score -= 0.4
    return max(0.0, min(1.0, score))


def run_benchmark(config: EngineConfig, prompts: Optional[List[Prompt]] = None,
                  engine: Optional[ConfigurableEngine] = None) -> AggregateResult:
    prompts = prompts or SUITE
    own_engine = engine is None
    eng = engine or ConfigurableEngine(config)
    eng.load()

    per_prompt: List[Dict] = []
    tps_vals, decode_vals, ttft_vals, lat_vals = [], [], [], []
    cpu_vals, gpu_vals, kv_vals, peak_vals = [], [], [], []
    quality_vals = []

    for p in prompts:
        # warmup (discarded)
        for _ in range(max(0, config.warmup_runs)):
            eng.generate(p)
        # measured runs
        runs: List[GenerationMetrics] = [eng.generate(p) for _ in range(max(1, config.measure_runs))]
        # pick median by tps for the representative record
        runs_sorted = sorted(runs, key=lambda m: m.tokens_per_second)
        rep = runs_sorted[len(runs_sorted) // 2]
        q = _quality_score(p, rep.output_text)
        quality_vals.append(q)

        tps_vals.append(statistics.median([m.tokens_per_second for m in runs]))
        decode_vals.append(statistics.median([m.decode_tokens_per_second for m in runs]))
        if rep.time_to_first_token_s is not None:
            ttft_vals.append(rep.time_to_first_token_s)
        lat_vals.append(statistics.median([m.total_latency_s for m in runs]))
        if rep.cpu_utilization_pct is not None:
            cpu_vals.append(rep.cpu_utilization_pct)
        if rep.gpu_utilization_pct is not None:
            gpu_vals.append(rep.gpu_utilization_pct)
        if rep.kv_cache_mb is not None:
            kv_vals.append(rep.kv_cache_mb)
        if rep.peak_memory_mb is not None:
            peak_vals.append(rep.peak_memory_mb)

        d = rep.as_dict()
        d["quality_score"] = round(q, 3)
        per_prompt.append(d)

    if own_engine:
        eng.unload()

    quality_score = round(sum(quality_vals) / len(quality_vals), 3) if quality_vals else 1.0
    agg = AggregateResult(
        label=config.label,
        mean_tps=round(statistics.mean(tps_vals), 3) if tps_vals else 0.0,
        median_tps=round(statistics.median(tps_vals), 3) if tps_vals else 0.0,
        mean_decode_tps=round(statistics.mean(decode_vals), 3) if decode_vals else 0.0,
        mean_ttft_s=round(statistics.mean(ttft_vals), 4) if ttft_vals else None,
        mean_latency_s=round(statistics.mean(lat_vals), 4) if lat_vals else 0.0,
        peak_memory_mb=round(max(peak_vals), 2) if peak_vals else None,
        mean_cpu_pct=round(statistics.mean(cpu_vals), 2) if cpu_vals else None,
        mean_gpu_pct=round(statistics.mean(gpu_vals), 2) if gpu_vals else None,
        kv_cache_mb=round(statistics.mean(kv_vals), 3) if kv_vals else None,
        quality_ok=quality_score >= 0.7,
        quality_score=quality_score,
        device=eng.device,
        per_prompt=per_prompt,
        support_notes=list(eng.support_notes),
    )
    return agg
