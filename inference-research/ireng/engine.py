"""engine.py — LlamaMoEEngine: GGUF-based MoE inference via llama-cpp-python.

This is the core engine the spec requires: it loads GGUF files, streams tokens,
measures telemetry (tok/s, TTFT, memory, expert stats), and exposes the same
knobs the experiment runner toggles (n_gpu_layers, use_mmap, flash_attn, etc.).

Expert selection tracking
-------------------------
llama-cpp-python does not expose per-layer router logits from Python.  Instead:
  1. GGUF metadata is parsed at load time to discover the expert architecture
     (n_experts, top_k, n_moe_layers).
  2. During generation we count the total decode steps and estimate bytes
     streamed per token from model size and top-K.
  3. A per-run ExpertStats object is populated with architecture info and the
     bytes-per-token estimate; per-step layer-level indices are marked as
     unavailable with a clear note rather than fabricated.
This is the most honest approach given the API constraints.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Iterator, Optional

from .config import EngineConfig, find_gguf_for_model
from .metrics import (
    GenerationMetrics, ExpertStats, ResourceSampler, Stopwatch,
)
from .prompts import Prompt


class EngineError(RuntimeError):
    pass


class LlamaMoEEngine:
    """Wraps llama-cpp-python Llama with MoE telemetry and experiment support."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._llm = None
        self._gguf_meta = None
        self._loaded = False
        self.support_notes: list[str] = []

    # ── Load / unload ────────────────────────────────────────────────────────

    def load(self) -> "LlamaMoEEngine":
        if self._loaded:
            return self

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise EngineError(
                "llama-cpp-python not installed.  "
                "Run: CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python"
            ) from e

        gguf_path = find_gguf_for_model(self.config.model_id)
        if not gguf_path:
            raise EngineError(
                f"No GGUF file found for model '{self.config.model_id}'.  "
                f"Download it first:\n"
                f"  hf download RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf "
                f"--include '*Q4_K_M*.gguf' "
                f"--local-dir ~/models/Qwen1.5-MoE-A2.7B-GGUF"
            )

        # Parse GGUF metadata before loading (fast, reads header only)
        try:
            from .gguf import read_gguf_meta
            self._gguf_meta = read_gguf_meta(gguf_path)
            if self._gguf_meta.is_moe:
                self.support_notes.append(
                    f"MoE model: {self._gguf_meta.n_experts} experts / layer, "
                    f"top-{self._gguf_meta.n_experts_used} active, "
                    f"{len(self._gguf_meta.moe_layers)} MoE layers"
                )
        except Exception as e:
            self.support_notes.append(f"GGUF metadata parse warning: {e}")

        kw = self.config.llama_kwargs()
        kw["model_path"] = gguf_path

        self._llm = Llama(**kw)
        self._loaded = True
        return self

    def unload(self):
        self._llm = None
        self._gguf_meta = None
        self._loaded = False
        gc.collect()

    def reconfigure(self, new_config: EngineConfig) -> "LlamaMoEEngine":
        """Reload with a new config (unloads current model first)."""
        self.unload()
        self.config = new_config
        self.support_notes = []
        return self.load()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _messages(self, prompt: Prompt) -> list[dict]:
        msgs = []
        if prompt.system:
            msgs.append({"role": "system", "content": prompt.system})
        msgs.append({"role": "user", "content": prompt.user})
        return msgs

    def _expert_stats(self, n_tokens: int) -> Optional[ExpertStats]:
        """Build ExpertStats from GGUF metadata + generation counts."""
        if not self._gguf_meta or not self._gguf_meta.is_moe:
            return None
        meta = self._gguf_meta
        stats = ExpertStats(
            n_experts=meta.n_experts,
            n_experts_used=meta.n_experts_used,
            n_moe_layers=len(meta.moe_layers),
        )
        # Per-step expert indices require llama.cpp C-level callbacks not
        # exposed by llama-cpp-python.  We document this honestly rather than
        # fabricating routing decisions.
        stats.per_step = []
        return stats

    def _expert_bytes_per_tok(self) -> Optional[float]:
        """Estimate bytes of expert weight read per token from GGUF size."""
        if not self._gguf_meta or not self._gguf_meta.is_moe:
            return None
        try:
            meta = self._gguf_meta
            file_gb  = os.path.getsize(meta.path) / 1024**3
            n_moe    = len(meta.moe_layers)
            n_total  = meta.n_layers
            # Expert weights are roughly (n_moe / n_total) fraction of file
            expert_gb   = file_gb * (n_moe / max(n_total, 1)) * 0.85
            expert_per_layer_gb = expert_gb / max(n_moe, 1)
            per_expert_gb = expert_per_layer_gb / max(meta.n_experts, 1)
            # Each token reads top-K experts per MoE layer
            bytes_per_tok = (per_expert_gb * meta.n_experts_used * n_moe
                             * 1024**3)
            return round(bytes_per_tok / 1024**2, 3)   # return in MB
        except Exception:
            return None

    # ── stream() ─────────────────────────────────────────────────────────────

    def stream(self, prompt: Prompt) -> Iterator[tuple[str, GenerationMetrics]]:
        """Yield (text_chunk, partial_metrics) as tokens arrive.

        The final yield is ("", complete_metrics) with all fields populated.
        """
        if not self._loaded:
            self.load()

        cfg = self.config
        messages = self._messages(prompt)
        sampler = ResourceSampler()
        sw = Stopwatch().start()
        sampler.start()

        n_tokens = 0
        output_parts: list[str] = []
        prompt_tokens = 0

        try:
            stream = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=prompt.max_new_tokens or cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                seed=cfg.seed,
                stream=True,
            )

            for chunk in stream:
                delta = (chunk.get("choices", [{}])[0]
                         .get("delta", {})
                         .get("content") or "")
                if delta:
                    sw.mark_first_token()
                    output_parts.append(delta)
                    n_tokens += 1

                elapsed = time.perf_counter() - sw.start_t
                partial = GenerationMetrics(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    model_id=cfg.model_id,
                    engine_label=cfg.label,
                    tokens_generated=n_tokens,
                    prompt_tokens=prompt_tokens,
                    context_length=prompt_tokens + n_tokens,
                    time_to_first_token_s=sw.ttft,
                    total_runtime_s=round(elapsed, 4),
                    tokens_per_second=round(n_tokens / max(elapsed, 1e-9), 3),
                    device="metal" if cfg.n_gpu_layers != 0 else "cpu",
                    output_text="".join(output_parts),
                )
                yield delta, partial

        except Exception as e:
            self.support_notes.append(f"generation error: {type(e).__name__}: {e}")

        sw.stop()
        sampler.stop()
        res = sampler.summary()
        total = sw.total or 1e-9

        # Get prompt token count from the model's tokenizer
        try:
            formatted = self._llm.tokenize(
                (prompt.system or "" + "\n" + prompt.user).encode())
            prompt_tokens = len(formatted)
        except Exception:
            prompt_tokens = 0

        expert_stats = self._expert_stats(n_tokens)
        expert_bytes = self._expert_bytes_per_tok()

        final = GenerationMetrics(
            prompt_id=prompt.id,
            category=prompt.category,
            model_id=cfg.model_id,
            engine_label=cfg.label,
            tokens_generated=n_tokens,
            prompt_tokens=prompt_tokens,
            context_length=prompt_tokens + n_tokens,
            time_to_first_token_s=sw.ttft,
            total_runtime_s=round(total, 4),
            tokens_per_second=round(n_tokens / total, 3),
            peak_memory_mb=res["peak_memory_mb"],
            current_memory_mb=res["current_memory_mb"],
            cpu_utilization_pct=res["cpu_utilization_pct"],
            gpu_utilization_pct=None,   # Metal GPU % not available from Python
            kv_cache_mb=None,           # llama.cpp KV cache not exposed to Python
            expert_bytes_per_tok=expert_bytes,
            page_cache_hit_rate=None,   # estimated in runner via warm/cold comparison
            expert_stats=expert_stats,
            device="metal" if cfg.n_gpu_layers != 0 else "cpu",
            output_text="".join(output_parts),
        )
        yield "", final

    # ── generate() ───────────────────────────────────────────────────────────

    def generate(self, prompt: Prompt) -> GenerationMetrics:
        """Non-streaming generation; returns the final metrics object."""
        last = None
        for _chunk, m in self.stream(prompt):
            last = m
        return last

    # ── Model info ───────────────────────────────────────────────────────────

    def model_summary(self) -> dict:
        if self._gguf_meta:
            return self._gguf_meta.summary()
        return {"model_id": self.config.model_id, "loaded": self._loaded}
