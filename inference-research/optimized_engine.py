"""optimized_engine.py — The highest-performing validated engine config.

Loads results/best_config.json (written by the experiment runner).
If that file doesn't exist yet (experiments not run), falls back to baseline.
"""
from __future__ import annotations

from ireng.config import EngineConfig, load_optimized_config, SMALL_MODEL_ID
from ireng.engine import LlamaMoEEngine


class OptimizedEngine(LlamaMoEEngine):
    def __init__(self, model_id: str = SMALL_MODEL_ID):
        cfg = load_optimized_config()
        cfg = EngineConfig.from_dict({**cfg.to_dict(), "model_id": model_id})
        super().__init__(cfg)


def load_optimized_config_for_model(model_id: str = SMALL_MODEL_ID) -> EngineConfig:
    cfg = load_optimized_config()
    return EngineConfig.from_dict({**cfg.to_dict(), "model_id": model_id})
