"""ConfigurableEngine — one engine, driven entirely by EngineConfig.

This single class implements every optimization knob the experiments toggle.
`baseline_engine.py` and `optimized_engine.py` are thin wrappers that hand it a
frozen config. Keeping all behaviour in one place is what lets 40 experiments
be a fair apples-to-apples comparison: only the config changes.

Design notes
------------
* torch / transformers are imported lazily so the framework (registry, storage,
  plotting, dashboard scaffolding) can be inspected and unit-tested on machines
  without a GPU or without the model downloaded.
* Unsupported knobs degrade gracefully and record a note rather than crash
  (e.g. bitsandbytes INT8 on Apple Silicon -> falls back + flags `supported=False`).
* Generation uses a TextIteratorStreamer so the dashboard can stream tokens and
  measure true time-to-first-token.
"""
from __future__ import annotations

import contextlib
import gc
import threading
from typing import Iterator, List, Optional, Tuple

from .config import EngineConfig
from .hardware import best_device, detect_host
from .metrics import GenerationMetrics, ResourceSampler, Stopwatch, kv_cache_mb
from .prompts import Prompt

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class EngineError(RuntimeError):
    pass


class ConfigurableEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = best_device(config.device)
        self.model = None
        self.tokenizer = None
        self.assistant_model = None
        self.support_notes: List[str] = []
        self._loaded = False

    # ------------------------------------------------------------------ load
    def load(self):
        if self._loaded:
            return self
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover - env without torch
            raise EngineError(f"torch/transformers unavailable: {e}")

        cfg = self.config

        if cfg.num_threads:
            torch.set_num_threads(int(cfg.num_threads))

        if cfg.matmul_precision in ("highest", "high", "medium"):
            try:
                torch.set_float32_matmul_precision(cfg.matmul_precision)
            except Exception as e:
                self.support_notes.append(f"matmul_precision unsupported: {e}")

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(cfg.dtype, torch.float32)

        # ---- tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, use_fast=cfg.use_fast_tokenizer
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- quantization config (graceful)
        quant_kwargs = self._build_quant_kwargs(cfg)

        load_kwargs = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        )
        # attention implementation
        if cfg.attn_implementation in ("sdpa", "eager", "flash_attention_2"):
            load_kwargs["attn_implementation"] = cfg.attn_implementation
        load_kwargs.update(quant_kwargs)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
        except Exception as e:
            # retry without optional knobs that some backends reject
            self.support_notes.append(f"load fallback (dropped optional kwargs): {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id, torch_dtype=dtype
            )

        if not quant_kwargs:  # quantized models are already device-placed
            self.model = self.model.to(self.device)

        if cfg.channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception as e:
                self.support_notes.append(f"channels_last unsupported: {e}")

        if cfg.eval_mode:
            self.model.eval()

        if cfg.attention_slicing:
            # Not all decoder models expose slicing; record support honestly.
            fn = getattr(self.model, "enable_attention_slicing", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    self.support_notes.append(f"attention_slicing failed: {e}")
            else:
                self.support_notes.append(
                    "attention_slicing not exposed by this model; "
                    "use sdpa mem_efficient backend (exp9) for the same goal.")

        if cfg.kv_cache_dtype:
            # HF does not expose a public per-layer KV dtype switch for all
            # models; we record it and rely on overall dtype / static cache.
            self.support_notes.append(
                f"kv_cache_dtype={cfg.kv_cache_dtype} requested; approximated via "
                f"model dtype + static cache where available.")

        # ---- torch.compile
        if cfg.torch_compile:
            try:
                self.model = torch.compile(self.model, mode=cfg.compile_mode)
            except Exception as e:
                self.support_notes.append(f"torch.compile unsupported: {e}")

        # ---- speculative / assistant model
        if cfg.speculative and cfg.assistant_model_id:
            try:
                self.assistant_model = AutoModelForCausalLM.from_pretrained(
                    cfg.assistant_model_id, torch_dtype=dtype
                ).to(self.device)
                if cfg.eval_mode:
                    self.assistant_model.eval()
            except Exception as e:
                self.support_notes.append(f"assistant model load failed: {e}")
                self.assistant_model = None

        self._loaded = True
        return self

    def _build_quant_kwargs(self, cfg: EngineConfig) -> dict:
        if cfg.quantization == "none":
            return {}
        host = detect_host()
        # bitsandbytes INT8/4-bit is CUDA-only. Flag clearly on Apple Silicon.
        if cfg.quant_backend == "bitsandbytes" and not host.has_cuda:
            self.support_notes.append(
                f"quantization={cfg.quantization} via bitsandbytes requires CUDA; "
                f"not supported on {self.device}. Running unquantized."
            )
            return {}
        try:
            from transformers import BitsAndBytesConfig

            if cfg.quantization == "int8":
                return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
            if cfg.quantization in ("nf4", "fp4"):
                import torch

                return {"quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=cfg.quantization,
                    bnb_4bit_compute_dtype=torch.float16,
                )}
        except Exception as e:
            self.support_notes.append(f"quantization unavailable: {e}")
        return {}

    # ----------------------------------------------------------- prompt prep
    def _format(self, prompt: Prompt) -> str:
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            sys = (prompt.system + "\n\n") if prompt.system else ""
            return f"{sys}{prompt.user}\n"

    @contextlib.contextmanager
    def _inference_ctx(self):
        import torch

        cfg = self.config
        if cfg.inference_mode:
            with torch.inference_mode():
                yield
        elif cfg.no_grad:
            with torch.no_grad():
                yield
        else:
            yield

    def _gen_kwargs(self, prompt: Prompt) -> dict:
        cfg = self.config
        base_max = prompt.max_new_tokens or cfg.max_new_tokens
        max_new = max(8, int(round(base_max * cfg.max_tokens_scale)))
        kw = dict(
            max_new_tokens=max_new,
            do_sample=cfg.do_sample,
            use_cache=cfg.use_cache,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if cfg.early_stop_eos and self.tokenizer.eos_token_id is not None:
            kw["eos_token_id"] = self.tokenizer.eos_token_id
        if cfg.do_sample:
            kw.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        if cfg.cache_implementation == "offloaded" and self.device == "mps":
            # OffloadedCache calls torch.cuda.default_stream() internally
            # which requires CUDA and is not available on MPS.
            self.support_notes.append(
                "cache_implementation='offloaded' requires CUDA; "
                "not supported on MPS. Running with dynamic cache."
            )
        elif cfg.cache_implementation in ("static", "offloaded"):
            kw["cache_implementation"] = cfg.cache_implementation
        if self.assistant_model is not None:
            kw["assistant_model"] = self.assistant_model
            kw["num_assistant_tokens"] = cfg.num_assistant_tokens
        return kw

    # --------------------------------------------------------------- stream
    def stream(self, prompt: Prompt) -> Iterator[Tuple[str, GenerationMetrics]]:
        """Yield (text_chunk, partial_metrics) as tokens are produced.

        The final yield carries the complete metrics object. Used by the
        dashboard for live streaming + true TTFT.
        """
        if not self._loaded:
            self.load()
        import torch
        from transformers import TextIteratorStreamer

        cfg = self.config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        text = self._format(prompt)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_tokens = int(enc["input_ids"].shape[1])

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = self._gen_kwargs(prompt)
        gen_kwargs.update(enc, streamer=streamer)

        sampler = ResourceSampler(self.device)
        sw = Stopwatch().start()
        sampler.start()

        thread_exc: list = []

        def _run():
            try:
                with self._inference_ctx():
                    self.model.generate(**gen_kwargs)
            except Exception as e:
                thread_exc.append(e)
                # Signal the streamer to unblock by ending the queue.
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        produced = []
        n_tokens = 0
        for chunk in streamer:
            if not chunk:
                continue
            sw.mark_first_token()
            produced.append(chunk)
            n_tokens += 1
            yield chunk, self._partial_metrics(prompt, prompt_tokens, n_tokens, sw, sampler)

        thread.join(timeout=30)
        sw.stop()
        sampler.stop()
        if thread_exc:
            exc = thread_exc[0]
            note = f"generation failed: {type(exc).__name__}: {exc}"
            self.support_notes.append(note)
        m = self._finalize(prompt, prompt_tokens, n_tokens, "".join(produced), sw, sampler)
        yield "", m

    def _partial_metrics(self, prompt, prompt_tokens, n, sw, sampler) -> GenerationMetrics:
        elapsed = sw.total if sw.end_t else (
            (sw.first_token_t or sw.start_t) and (__import__("time").perf_counter() - sw.start_t)
        )
        elapsed = elapsed or 1e-9
        return GenerationMetrics(
            prompt_id=prompt.id, category=prompt.category,
            tokens_generated=n, prompt_tokens=prompt_tokens,
            context_length=prompt_tokens + n,
            time_to_first_token_s=sw.ttft,
            total_latency_s=round(elapsed, 4),
            tokens_per_second=round(n / elapsed, 3),
            device=self.device,
        )

    def _finalize(self, prompt, prompt_tokens, n, output, sw, sampler) -> GenerationMetrics:
        res = sampler.summary()
        total = sw.total or 1e-9
        decode_time = (total - (sw.ttft or 0.0)) or 1e-9
        dbytes = _DTYPE_BYTES.get(self.config.dtype, 4)
        m = GenerationMetrics(
            prompt_id=prompt.id, category=prompt.category,
            tokens_generated=n, prompt_tokens=prompt_tokens,
            context_length=prompt_tokens + n,
            time_to_first_token_s=round(sw.ttft, 4) if sw.ttft else None,
            total_latency_s=round(total, 4),
            tokens_per_second=round(n / total, 3),
            decode_tokens_per_second=round(max(n - 1, 0) / decode_time, 3),
            peak_memory_mb=res["peak_memory_mb"],
            current_memory_mb=res["current_memory_mb"],
            cpu_utilization_pct=res["cpu_utilization_pct"],
            gpu_utilization_pct=res["gpu_utilization_pct"],
            kv_cache_mb=kv_cache_mb(self.model.config, prompt_tokens, n, dbytes),
            device=self.device,
            output_text=output,
        )
        return m

    # ---------------------------------------------------------- batch / once
    def generate(self, prompt: Prompt) -> GenerationMetrics:
        """Non-streaming generation, returns final metrics. Median over
        config.measure_runs is handled by the benchmark runner, not here."""
        last = None
        for _chunk, m in self.stream(prompt):
            last = m
        return last

    def unload(self):
        self.model = None
        self.assistant_model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        try:
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
