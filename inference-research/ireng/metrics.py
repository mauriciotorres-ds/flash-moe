"""Metrics collection for inference benchmarking.

Captures everything the spec asks for:
  tokens generated, tokens/sec, time-to-first-token, total latency,
  peak & current memory, CPU utilisation, GPU utilisation, KV cache size,
  context length.

All values come from real execution (psutil + torch). Where a metric cannot
be obtained on a given backend (e.g. GPU utilisation on Apple MPS), it is
reported as ``None`` and labelled accordingly — never fabricated.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class GenerationMetrics:
    prompt_id: str = ""
    category: str = ""
    tokens_generated: int = 0
    prompt_tokens: int = 0
    context_length: int = 0
    time_to_first_token_s: Optional[float] = None
    total_latency_s: float = 0.0
    tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0   # excludes prefill/TTFT
    peak_memory_mb: Optional[float] = None
    current_memory_mb: Optional[float] = None
    cpu_utilization_pct: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    kv_cache_mb: Optional[float] = None
    device: str = ""
    output_text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceSampler:
    """Background sampler for CPU%, RSS memory, and (when available) GPU%.

    Usage:
        s = ResourceSampler(device="mps"); s.start()
        ... run generation ...
        s.stop(); summary = s.summary()
    """

    def __init__(self, device: str = "cpu", interval: float = 0.1):
        self.device = device
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cpu_samples: List[float] = []
        self.mem_samples_mb: List[float] = []
        self.gpu_samples: List[float] = []
        self._proc = None
        try:
            import psutil

            self._proc = psutil.Process()
            self._proc.cpu_percent(None)  # prime the first reading
        except Exception:
            self._proc = None

    def _loop(self):
        from .hardware import gpu_utilization

        while not self._stop.is_set():
            if self._proc is not None:
                try:
                    self.cpu_samples.append(self._proc.cpu_percent(None))
                    self.mem_samples_mb.append(self._proc.memory_info().rss / 1e6)
                except Exception:
                    pass
            g = gpu_utilization(self.device)
            if g is not None:
                self.gpu_samples.append(g)
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, Optional[float]]:
        def _avg(xs):
            return round(sum(xs) / len(xs), 2) if xs else None

        def _max(xs):
            return round(max(xs), 2) if xs else None

        return {
            "cpu_utilization_pct": _avg(self.cpu_samples),
            "current_memory_mb": round(self.mem_samples_mb[-1], 2) if self.mem_samples_mb else None,
            "peak_memory_mb": _max(self.mem_samples_mb),
            "gpu_utilization_pct": _avg(self.gpu_samples),  # None on MPS/CPU
        }


def kv_cache_mb(model_config, prompt_tokens: int, generated_tokens: int, dtype_bytes: int = 4) -> Optional[float]:
    """Estimate KV-cache size in MB from model architecture.

    size = 2 (K+V) * layers * heads_kv * head_dim * seq_len * dtype_bytes
    Uses GQA-aware num_key_value_heads when present.
    """
    try:
        layers = getattr(model_config, "num_hidden_layers")
        hidden = getattr(model_config, "hidden_size")
        n_heads = getattr(model_config, "num_attention_heads")
        n_kv = getattr(model_config, "num_key_value_heads", n_heads)
        head_dim = hidden // n_heads
        seq = prompt_tokens + generated_tokens
        total = 2 * layers * n_kv * head_dim * seq * dtype_bytes
        return round(total / 1e6, 3)
    except Exception:
        return None


class Stopwatch:
    """Tiny monotonic timer with a first-token marker for TTFT."""

    def __init__(self):
        self.start_t = None
        self.first_token_t = None
        self.end_t = None

    def start(self):
        self.start_t = time.perf_counter()
        return self

    def mark_first_token(self):
        if self.first_token_t is None:
            self.first_token_t = time.perf_counter()

    def stop(self):
        self.end_t = time.perf_counter()

    @property
    def ttft(self) -> Optional[float]:
        if self.start_t is None or self.first_token_t is None:
            return None
        return self.first_token_t - self.start_t

    @property
    def total(self) -> float:
        if self.start_t is None or self.end_t is None:
            return 0.0
        return self.end_t - self.start_t
