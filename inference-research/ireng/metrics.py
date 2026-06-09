"""metrics.py — Telemetry for MoE inference benchmarking.

Per spec, every generation records:
  - total_runtime_s       end-to-end wall-clock time
  - tokens_per_second     tokens ÷ total_runtime_s
  - expert_selection      per-step, per-layer top-K expert indices
  - expert_bytes_per_tok  bytes of expert weights read per token (streaming)
  - page_cache_hit_rate   OS page cache hit rate for expert reads (estimated)
  - Plus: TTFT, peak memory, CPU%, context length, etc.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ── Expert selection record ───────────────────────────────────────────────────

@dataclass
class ExpertSelectionStep:
    """Router decisions for one decoding step across all MoE layers."""
    step: int
    # layer_idx → list of top-K expert indices chosen
    layer_experts: dict[int, list[int]] = field(default_factory=dict)
    # layer_idx → list of routing weights (probabilities), if available
    layer_weights: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class ExpertStats:
    """Aggregated expert utilisation across an entire generation."""
    n_experts:      int = 0
    n_experts_used: int = 0   # top-K
    n_moe_layers:   int = 0
    # expert_id → number of times selected across all steps and layers
    activation_counts: dict[int, int] = field(default_factory=dict)
    # Available only when per-step data was captured
    per_step: list[ExpertSelectionStep] = field(default_factory=list)

    def top_experts(self, n: int = 10) -> list[tuple[int, int]]:
        sorted_experts = sorted(self.activation_counts.items(),
                                key=lambda x: x[1], reverse=True)
        return sorted_experts[:n]

    def utilization_rate(self) -> float:
        """Fraction of all experts that were ever activated."""
        if not self.n_experts:
            return 0.0
        return len(self.activation_counts) / self.n_experts

    def to_dict(self) -> dict:
        return {
            "n_experts": self.n_experts,
            "n_experts_used": self.n_experts_used,
            "n_moe_layers": self.n_moe_layers,
            "activation_counts": self.activation_counts,
            "utilization_rate": round(self.utilization_rate(), 4),
            "top_10_experts": self.top_experts(10),
        }


# ── Main result dataclass ─────────────────────────────────────────────────────

@dataclass
class GenerationMetrics:
    prompt_id:   str = ""
    category:    str = ""
    model_id:    str = ""
    engine_label: str = ""

    # Token counts
    tokens_generated: int = 0
    prompt_tokens:    int = 0
    context_length:   int = 0

    # Timing  (spec: total_runtime_s is mandatory)
    total_runtime_s:       float          = 0.0
    time_to_first_token_s: Optional[float] = None
    tokens_per_second:     float          = 0.0

    # Memory
    peak_memory_mb:    Optional[float] = None
    current_memory_mb: Optional[float] = None

    # Hardware utilisation
    cpu_utilization_pct: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None   # None on MPS

    # KV cache
    kv_cache_mb: Optional[float] = None

    # Expert streaming  (spec: expert_selection is mandatory)
    expert_bytes_per_tok:  Optional[float] = None   # avg bytes read per token
    page_cache_hit_rate:   Optional[float] = None   # 0–1
    expert_stats: Optional[ExpertStats]    = None

    output_text: str = ""
    device:      str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Flatten expert_stats for CSV/JSON storage
        if self.expert_stats:
            d["expert_stats"] = self.expert_stats.to_dict()
        return d

    # Alias kept for dashboard compatibility
    @property
    def total_latency_s(self) -> float:
        return self.total_runtime_s

    @property
    def mean_tps(self) -> float:
        return self.tokens_per_second


# ── Background resource sampler ───────────────────────────────────────────────

class ResourceSampler:
    """Samples CPU%, RSS, and (where available) GPU% in a background thread."""

    def __init__(self, interval: float = 0.15):
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.interval = interval
        self.cpu_samples:    list[float] = []
        self.mem_samples_mb: list[float] = []
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)   # prime the counter
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            if self._proc:
                try:
                    self.cpu_samples.append(self._proc.cpu_percent(None))
                    self.mem_samples_mb.append(
                        self._proc.memory_info().rss / 1_048_576)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Optional[float]]:
        def _avg(xs): return round(sum(xs) / len(xs), 2) if xs else None
        def _max(xs): return round(max(xs), 2) if xs else None
        return {
            "cpu_utilization_pct": _avg(self.cpu_samples),
            "current_memory_mb":   round(self.mem_samples_mb[-1], 2) if self.mem_samples_mb else None,
            "peak_memory_mb":      _max(self.mem_samples_mb),
        }


# ── Stopwatch ─────────────────────────────────────────────────────────────────

class Stopwatch:
    def __init__(self):
        self.start_t:       Optional[float] = None
        self.first_token_t: Optional[float] = None
        self.end_t:         Optional[float] = None

    def start(self) -> "Stopwatch":
        self.start_t = time.perf_counter()
        return self

    def mark_first_token(self):
        if self.first_token_t is None:
            self.first_token_t = time.perf_counter()

    def stop(self):
        self.end_t = time.perf_counter()

    @property
    def ttft(self) -> Optional[float]:
        if self.start_t and self.first_token_t:
            return round(self.first_token_t - self.start_t, 4)
        return None

    @property
    def total(self) -> float:
        if self.start_t and self.end_t:
            return round(self.end_t - self.start_t, 4)
        return 0.0
