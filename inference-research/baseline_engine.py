#!/usr/bin/env python3
"""baseline_engine.py — the reproducible reference inference engine.

Plain Hugging Face Transformers, float32, eager attention, dynamic KV cache,
no runtime tricks. This is the control that every experiment is measured
against. It is a thin wrapper over ireng.ConfigurableEngine with the frozen
baseline config so there is exactly one engine implementation to trust.

Usage:
  python baseline_engine.py --prompt "Explain MoE in one sentence." --max-new-tokens 96
  python baseline_engine.py --suite          # run the whole benchmark suite
"""
from __future__ import annotations

import argparse
import json

from ireng.config import baseline_config
from ireng.engine import ConfigurableEngine
from ireng.prompts import Prompt, SUITE
from ireng.hardware import detect_host, TARGET_SPEC


class BaselineEngine(ConfigurableEngine):
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="auto"):
        super().__init__(baseline_config(model_id).delta(device=device))


def _run_one(engine, system, user, max_new_tokens):
    p = Prompt("cli", "factual", system, user, max_new_tokens)
    print(f"\n--- generating ({engine.device}) ---")
    text = []
    for chunk, m in engine.stream(p):
        if chunk:
            print(chunk, end="", flush=True)
            text.append(chunk)
    print("\n--- metrics ---")
    print(json.dumps({k: v for k, v in m.as_dict().items() if k != "output_text"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt", default="Explain Mixture-of-Experts in one sentence.")
    ap.add_argument("--system", default="You are a concise, helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--suite", action="store_true", help="run the full benchmark suite")
    a = ap.parse_args()

    print(f"Target: {TARGET_SPEC['label']}")
    host = detect_host()
    print(f"Host device: {host.device}  (torch={host.torch_version})")

    engine = BaselineEngine(a.model, a.device)
    engine.load()
    if a.suite:
        for p in SUITE:
            _run_one(engine, p.system, p.user, p.max_new_tokens)
    else:
        _run_one(engine, a.system, a.prompt, a.max_new_tokens)
    engine.unload()


if __name__ == "__main__":
    main()
