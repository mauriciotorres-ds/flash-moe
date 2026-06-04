#!/usr/bin/env python3
"""optimized_engine.py — the highest-performing *validated* configuration.

It loads `results/best_config.json` (produced by the autoresearch runner) and
builds the same ConfigurableEngine from it. If that file does not exist yet
(you haven't run the experiments on your Mac), it falls back to a sensible,
documented default: inference_mode + fp16 + SDPA + static cache + greedy decode
— the optimizations that are reliable wins on Apple Silicon MPS.

The fallback default is clearly labelled as such; the *authoritative* optimized
engine is whatever the measured experiments selected.

Usage:
  python optimized_engine.py --prompt "Explain MoE in one sentence."
  python optimized_engine.py --suite
"""
from __future__ import annotations

import argparse
import json
import os

from ireng.config import EngineConfig, baseline_config, diff_configs
from ireng.engine import ConfigurableEngine
from ireng.prompts import Prompt, SUITE
from ireng.hardware import detect_host, TARGET_SPEC
from ireng import storage as st


def _fallback_optimized(model_id: str) -> EngineConfig:
    return baseline_config(model_id).delta(
        inference_mode=True,
        dtype="float16",
        attn_implementation="sdpa",
        cache_implementation="static",
        do_sample=False,
        matmul_precision="high",
        warmup_runs=2,
        label="optimized(default-fallback)",
        notes="Default optimized config used until measured best_config.json exists.",
    )


def load_optimized_config(model_id="Qwen/Qwen2.5-0.5B-Instruct", device="auto") -> EngineConfig:
    bc = st.read_json(st.BEST_CONFIG_JSON, default=None)
    if bc and bc.get("config"):
        cfg = EngineConfig(**{k: v for k, v in bc["config"].items()
                              if k in EngineConfig.__dataclass_fields__})
        cfg = cfg.delta(device=device, label=f"optimized({bc.get('label','best')})")
        return cfg
    return _fallback_optimized(model_id).delta(device=device)


class OptimizedEngine(ConfigurableEngine):
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="auto"):
        super().__init__(load_optimized_config(model_id, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt", default="Explain Mixture-of-Experts in one sentence.")
    ap.add_argument("--system", default="You are a concise, helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--show-config", action="store_true")
    a = ap.parse_args()

    print(f"Target: {TARGET_SPEC['label']}")
    cfg = load_optimized_config(a.model, a.device)
    print(f"Optimized config: {cfg.label}")
    if a.show_config:
        print(json.dumps(cfg.signature(), indent=2))
        print("\nDiff vs baseline:")
        print(json.dumps(diff_configs(baseline_config(a.model), cfg), indent=2))

    engine = OptimizedEngine(a.model, a.device)
    engine.load()
    prompts = SUITE if a.suite else [Prompt("cli", "factual", a.system, a.prompt, a.max_new_tokens)]
    for p in prompts:
        print(f"\n--- {p.id} ({engine.device}) ---")
        last = None
        for chunk, m in engine.stream(p):
            if chunk:
                print(chunk, end="", flush=True)
            last = m
        print("\n" + json.dumps(
            {k: v for k, v in last.as_dict().items() if k != "output_text"}, indent=2))
    engine.unload()


if __name__ == "__main__":
    main()
