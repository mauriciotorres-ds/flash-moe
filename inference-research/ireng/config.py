"""config.py — EngineConfig and model-path resolution.

All llama-cpp-python knobs live here.  Each experiment changes one or more
fields; the engine reloads with the new config so results are reproducible.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

# ── Model directories (kept outside the repo) ─────────────────────────────────
_MODEL_ROOT = os.path.expanduser("~/models")

SMALL_MODEL_ID  = "Qwen1.5-MoE-A2.7B"
MEDIUM_MODEL_ID = "Qwen3.5-35B-A3B"

SMALL_GGUF_DIR  = os.path.join(_MODEL_ROOT, "Qwen1.5-MoE-A2.7B-GGUF")
MEDIUM_GGUF_DIR = os.path.join(_MODEL_ROOT, "Qwen3.5-35B-A3B-GGUF")

MODEL_DIRS = {
    SMALL_MODEL_ID:  SMALL_GGUF_DIR,
    MEDIUM_MODEL_ID: MEDIUM_GGUF_DIR,
}


def find_gguf_for_model(model_id: str) -> str | None:
    from ireng.gguf import find_gguf
    return find_gguf(MODEL_DIRS.get(model_id, ""))


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    """All tuneable parameters for one experiment run."""

    label: str = "baseline"

    # Model
    model_id: str = SMALL_MODEL_ID

    # Context / batch
    n_ctx:   int = 2048
    n_batch: int = 512

    # CPU threading  (-1 = os.cpu_count())
    n_threads:       int = -1
    n_threads_batch: int = -1

    # GPU offloading via Metal  (0 = CPU-only, -1 = all layers on GPU)
    n_gpu_layers: int = 0

    # Memory mapping
    use_mmap:  bool = True
    use_mlock: bool = False

    # Flash attention
    flash_attn: bool = False

    # Generation defaults (overridden per-prompt at call time)
    max_new_tokens: int   = 128
    temperature:    float = 0.0
    top_p:          float = 1.0
    top_k:          int   = 1
    seed:           int   = 1234

    # Provenance
    notes: str = ""

    def resolved_n_threads(self) -> int:
        return self.n_threads if self.n_threads > 0 else (os.cpu_count() or 4)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})

    def llama_kwargs(self) -> dict:
        """Keyword arguments to pass directly to Llama(...)."""
        n_t = self.resolved_n_threads()
        return dict(
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=n_t,
            n_threads_batch=(self.n_threads_batch if self.n_threads_batch > 0 else n_t),
            n_gpu_layers=self.n_gpu_layers,
            use_mmap=self.use_mmap,
            use_mlock=self.use_mlock,
            flash_attn=self.flash_attn,
            verbose=False,
        )


# ── Preset constructors ───────────────────────────────────────────────────────

def baseline_config() -> EngineConfig:
    return EngineConfig(
        label="baseline",
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False,
        flash_attn=False,
        n_ctx=2048,
        n_batch=512,
        notes="Naive baseline: mmap on, CPU-only, no flash-attn, trust OS page cache.",
    )


def load_optimized_config() -> EngineConfig:
    """Load winning config from results/best_config.json, or return baseline."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "results", "best_config.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        try:
            return EngineConfig.from_dict(d.get("config", d))
        except Exception:
            pass
    return baseline_config()


def diff_configs(base: EngineConfig, opt: EngineConfig) -> dict:
    bd, od = base.to_dict(), opt.to_dict()
    return {k: {"baseline": bd[k], "optimized": od[k]}
            for k in bd if bd[k] != od[k]}
