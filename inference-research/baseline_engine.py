"""baseline_engine.py — Naive GGUF MoE baseline engine.

Loads Qwen1.5-MoE-A2.7B Q4_K_M via mmap (trust OS page cache),
CPU-only, no flash attention, default batch size.
This is exp000 — the reference every optimization is measured against.
"""
from __future__ import annotations

from ireng.config import EngineConfig, baseline_config, SMALL_MODEL_ID
from ireng.engine import LlamaMoEEngine


class BaselineEngine(LlamaMoEEngine):
    def __init__(self, model_id: str = SMALL_MODEL_ID):
        cfg = baseline_config()
        cfg = EngineConfig.from_dict({**cfg.to_dict(), "model_id": model_id})
        super().__init__(cfg)


def load_baseline_config() -> EngineConfig:
    return baseline_config()
