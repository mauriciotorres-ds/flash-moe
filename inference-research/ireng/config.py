"""EngineConfig — the single source of truth for every optimization knob.

Every experiment is expressed as a *delta* on top of a base config. The
configurable engine (`ireng/engine.py`) reads this object and nothing else, so
"running experiment N" means "build EngineConfig from baseline + exp-N delta,
benchmark it, compare to current best". This is what makes 40 experiments
reproducible and what makes `optimized_engine.py` simply a frozen config.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace, field
from typing import Optional, Dict, Any


@dataclass
class EngineConfig:
    # ---- model / device -------------------------------------------------
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"            # auto | mps | cuda | cpu
    dtype: str = "float32"          # float32 | float16 | bfloat16

    # ---- runtime --------------------------------------------------------
    inference_mode: bool = False    # torch.inference_mode() context
    no_grad: bool = False           # torch.no_grad() context (subset of above)
    eval_mode: bool = True          # model.eval()
    torch_compile: bool = False
    compile_mode: str = "default"   # default | reduce-overhead | max-autotune
    num_threads: Optional[int] = None   # torch.set_num_threads (CPU paths)
    channels_last: bool = False

    # ---- attention / cache ---------------------------------------------
    attn_implementation: str = "eager"   # eager | sdpa | flash_attention_2
    sdpa_backend: str = "auto"           # auto | math | flash | mem_efficient
    use_cache: bool = True
    cache_implementation: str = "dynamic"  # dynamic | static | offloaded
    static_cache_max_len: int = 2048

    # ---- model loading --------------------------------------------------
    low_cpu_mem_usage: bool = False
    pin_memory: bool = False

    # ---- quantization ---------------------------------------------------
    quantization: str = "none"      # none | int8 | nf4 | fp4
    quant_backend: str = "none"     # none | bitsandbytes | torchao

    # ---- tokenizer ------------------------------------------------------
    use_fast_tokenizer: bool = True

    # ---- generation -----------------------------------------------------
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    seed: int = 1234
    batch_size: int = 1            # >1 enables prompt batching / throughput mode

    # ---- decoding strategies -------------------------------------------
    speculative: bool = False
    assistant_model_id: Optional[str] = None   # draft model for spec decoding
    num_assistant_tokens: int = 5
    early_stop_eos: bool = True

    # ---- finer-grained knobs (exp33-40) --------------------------------
    matmul_precision: str = "highest"      # highest | high | medium
    kv_cache_dtype: Optional[str] = None   # None | float16 | bfloat16
    attention_slicing: bool = False
    max_tokens_scale: float = 1.0          # length-adaptive generation cap
    reuse_generation_config: bool = False
    batch_tokenize: bool = False

    # ---- benchmark behaviour -------------------------------------------
    warmup_runs: int = 1
    measure_runs: int = 3          # measured repetitions per prompt (median)

    # ---- provenance -----------------------------------------------------
    label: str = "baseline"
    notes: str = ""

    def delta(self, **kwargs) -> "EngineConfig":
        """Return a copy with the given fields overridden."""
        return replace(self, **kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def signature(self) -> Dict[str, Any]:
        """The subset that actually affects measured performance (for diffing)."""
        keys = [
            "dtype", "device", "inference_mode", "no_grad", "eval_mode",
            "torch_compile", "compile_mode", "num_threads", "channels_last",
            "attn_implementation", "sdpa_backend", "use_cache",
            "cache_implementation", "low_cpu_mem_usage", "pin_memory",
            "quantization", "quant_backend", "use_fast_tokenizer",
            "do_sample", "batch_size", "speculative", "assistant_model_id",
        ]
        return {k: getattr(self, k) for k in keys}


def baseline_config(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> EngineConfig:
    """The reproducible *baseline*: plain HF Transformers, float32, eager attention,
    dynamic cache, no runtime tricks. This is exp000 / the reference point."""
    return EngineConfig(
        model_id=model_id,
        device="auto",
        dtype="float32",
        inference_mode=False,
        no_grad=False,
        eval_mode=True,
        torch_compile=False,
        attn_implementation="eager",
        use_cache=True,
        cache_implementation="dynamic",
        use_fast_tokenizer=True,
        label="baseline",
        notes="Reference HF Transformers baseline. No optimizations applied.",
    )


def diff_configs(a: EngineConfig, b: EngineConfig) -> Dict[str, Any]:
    """Field-level diff used by the dashboard's Optimization Diff Viewer."""
    da, db = a.as_dict(), b.as_dict()
    out: Dict[str, Any] = {}
    for k in da:
        if da[k] != db.get(k):
            out[k] = {"baseline": da[k], "optimized": db.get(k)}
    return out
