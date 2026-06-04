"""The experiment registry — 42 runnable experiments (spec requires >= 40).

Each experiment is a *delta* of config knobs plus research metadata. It is NOT
a result: the keep/discard decision is produced by the runner from real
measurements on the host (Apple M4 target). `base` controls whether the delta
is applied on top of the current best config ("best", the autoresearch default
— experiments build on prior wins) or on the clean baseline ("baseline", for
isolating a single knob's effect).

Categories mirror the spec: runtime, memory, quantization, scheduling,
decoding, system, ssd_streaming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass(frozen=True)
class Experiment:
    exp: int
    title: str
    category: str
    base: str                 # "best" | "baseline"
    overrides: Dict[str, Any]
    hypothesis: str
    rationale: str            # which bottleneck it targets / why now
    prior: str = ""           # expectation (a prior, never a fabricated result)
    expect_quality_risk: bool = False
    expect_unsupported_on: str = ""   # e.g. "mps" — documents likely fallback


REGISTRY: List[Experiment] = [
    Experiment(1, "torch.no_grad() context", "runtime", "baseline",
        {"no_grad": True, "label": "no_grad"},
        "Disabling autograd bookkeeping reduces per-step overhead.",
        "Generation never needs gradients; baseline leaves autograd active.",
        prior="Small but free win on most backends."),

    Experiment(2, "torch.inference_mode() context", "runtime", "best",
        {"inference_mode": True, "no_grad": False, "label": "inference_mode"},
        "inference_mode is stricter than no_grad and skips version counters.",
        "Targets per-token Python/ATen overhead, the dominant cost at 0.5B.",
        prior="Usually >= no_grad; expected to replace it."),

    Experiment(3, "model.eval() isolation", "runtime", "baseline",
        {"eval_mode": False, "label": "no_eval(ablation)"},
        "Leaving train mode on enables dropout/other train-time paths.",
        "Ablation to confirm eval() matters and the harness detects it.",
        prior="eval() expected better; this run should be slower/worse quality.",
        expect_quality_risk=True),

    Experiment(4, "float16 weights", "quantization", "best",
        {"dtype": "float16", "label": "fp16"},
        "Half precision halves memory traffic; MPS has fast fp16 paths.",
        "Memory bandwidth is the bottleneck once Python overhead is cut.",
        prior="Often the single biggest win on Apple GPUs."),

    Experiment(5, "bfloat16 weights", "quantization", "baseline",
        {"dtype": "bfloat16", "label": "bf16"},
        "bf16 keeps fp32 dynamic range with fp16 bandwidth.",
        "Compare against fp16 for a quality/throughput trade.",
        prior="Throughput similar to fp16; possibly better numerics."),

    Experiment(6, "SDPA attention", "runtime", "best",
        {"attn_implementation": "sdpa", "label": "sdpa"},
        "Fused scaled-dot-product-attention beats eager Python attention.",
        "Attention is a per-layer cost; eager is the naive reference.",
        prior="Reliable win where supported."),

    Experiment(7, "FlashAttention-2", "runtime", "best",
        {"attn_implementation": "flash_attention_2", "label": "flash_attn2"},
        "FA2 reduces memory movement in attention.",
        "Test whether FA2 kernels are usable on this host.",
        prior="Expected UNSUPPORTED on MPS -> graceful fallback (a failure to log).",
        expect_unsupported_on="mps"),

    Experiment(8, "SDPA math backend", "runtime", "best",
        {"sdpa_backend": "math", "label": "sdpa_math"},
        "Forcing the math kernel can be faster for tiny head dims.",
        "Probe which SDPA backend the scheduler should pin.",
        prior="Backend-dependent; measure."),

    Experiment(9, "SDPA mem-efficient backend", "runtime", "best",
        {"sdpa_backend": "mem_efficient", "label": "sdpa_memeff"},
        "Memory-efficient attention lowers peak memory.",
        "Targets memory bottleneck for longer contexts.",
        prior="May trade a little speed for memory."),

    Experiment(10, "use_cache ablation", "memory", "baseline",
        {"use_cache": False, "label": "no_kv_cache(ablation)"},
        "Disabling the KV cache forces full recompute each step.",
        "Ablation: quantify how much the KV cache is worth.",
        prior="Expected MUCH slower; validates cache value.",),

    Experiment(11, "Static KV cache", "memory", "best",
        {"cache_implementation": "static", "label": "static_cache"},
        "Pre-allocated static cache avoids per-step reallocation and is "
        "required for clean torch.compile graphs.",
        "Removes allocation churn; enables later compile combo.",
        prior="Small win alone; unlocks exp29 combo."),

    Experiment(12, "Offloaded KV cache", "memory", "best",
        {"cache_implementation": "offloaded", "label": "offloaded_cache"},
        "Offloading cache to CPU frees GPU memory for weights.",
        "Targets the 24GB memory ceiling for long contexts.",
        prior="Likely slower for short prompts; helps long-context only.",),

    Experiment(13, "torch.compile (default)", "runtime", "best",
        {"torch_compile": True, "compile_mode": "default", "label": "compile_default"},
        "Graph capture fuses ops and cuts dispatch overhead.",
        "Per-token dispatch overhead is large for small models.",
        prior="First-call compile cost; warmup hides it."),

    Experiment(14, "torch.compile (reduce-overhead)", "runtime", "best",
        {"torch_compile": True, "compile_mode": "reduce-overhead", "label": "compile_reduce_ovh"},
        "CUDA-graph-style overhead reduction for the decode loop.",
        "Decode loop is launch-bound; reduce-overhead targets exactly that.",
        prior="Often best compile mode for autoregressive decode."),

    Experiment(15, "torch.compile (max-autotune)", "runtime", "best",
        {"torch_compile": True, "compile_mode": "max-autotune", "label": "compile_max_autotune"},
        "Autotuned kernels maximise throughput.",
        "Spend compile time to win steady-state tok/s.",
        prior="Long compile; may or may not beat reduce-overhead."),

    Experiment(16, "low_cpu_mem_usage loading", "system", "best",
        {"low_cpu_mem_usage": True, "label": "low_cpu_mem_load"},
        "Streamed weight loading lowers peak RAM during init.",
        "Model-loading bottleneck + 24GB ceiling.",
        prior="Affects load time/peak RAM, not steady tok/s."),

    Experiment(17, "channels_last memory format", "memory", "best",
        {"channels_last": True, "label": "channels_last"},
        "channels_last can improve some kernel memory access.",
        "Cheap to try; targets memory layout.",
        prior="Usually neutral for transformers; measure.",),

    Experiment(18, "Pin threads to performance cores", "system", "baseline",
        {"num_threads": 4, "device": "cpu", "label": "cpu_4threads_pcores"},
        "On CPU paths, limiting to the 4 P-cores avoids E-core contention.",
        "System scheduling bottleneck on the M4's hybrid CPU.",
        prior="Only relevant if a CPU fallback path is used."),

    Experiment(19, "Slow (Python) tokenizer ablation", "runtime", "baseline",
        {"use_fast_tokenizer": False, "label": "slow_tokenizer(ablation)"},
        "The Rust fast tokenizer should beat the Python one at encode.",
        "Tokenizer/startup bottleneck (cf. Flash-MoE's C BPE, 20x startup).",
        prior="Fast tokenizer expected better; this run should regress.",),

    Experiment(20, "INT8 (bitsandbytes)", "quantization", "best",
        {"quantization": "int8", "quant_backend": "bitsandbytes", "label": "int8_bnb"},
        "8-bit weights cut memory traffic ~4x vs fp32.",
        "Bandwidth bottleneck; biggest theoretical memory win.",
        prior="bitsandbytes is CUDA-only -> expected UNSUPPORTED on MPS.",
        expect_unsupported_on="mps"),

    Experiment(21, "NF4 4-bit (bitsandbytes)", "quantization", "best",
        {"quantization": "nf4", "quant_backend": "bitsandbytes", "label": "nf4_bnb"},
        "4-bit weights for maximum memory reduction.",
        "Mirrors Flash-MoE's 4-bit expert packing.",
        prior="CUDA-only; expected UNSUPPORTED on MPS. Quality risk if forced.",
        expect_unsupported_on="mps", expect_quality_risk=True),

    Experiment(22, "Prompt batching (bs=4)", "scheduling", "best",
        {"batch_size": 4, "label": "batch4"},
        "Batching amortises kernel launches across prompts.",
        "Throughput scheduling: improves aggregate tok/s under load.",
        prior="Improves throughput, not single-stream latency."),

    Experiment(23, "Prompt batching (bs=8)", "scheduling", "best",
        {"batch_size": 8, "label": "batch8"},
        "Larger batch -> higher utilisation until memory-bound.",
        "Find the batch size that saturates the 10-core GPU.",
        prior="Diminishing returns / memory pressure past some point.",),

    Experiment(24, "Speculative decoding (draft model)", "decoding", "best",
        {"speculative": True, "assistant_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
         "num_assistant_tokens": 5, "label": "speculative"},
        "A small draft proposes tokens the target verifies in parallel.",
        "Decode is latency-bound; speculation can raise effective tok/s.",
        prior="Gains depend on draft acceptance rate; can break even.",),

    Experiment(25, "Greedy vs sampling cost", "decoding", "best",
        {"do_sample": False, "label": "greedy_decode"},
        "Greedy avoids sampling/softmax-top-k overhead.",
        "Isolate sampler cost from model compute.",
        prior="Greedy slightly faster; default decode for benchmarking."),

    Experiment(26, "Early EOS stopping", "decoding", "best",
        {"early_stop_eos": True, "label": "early_eos"},
        "Stop as soon as EOS is produced instead of padding to max.",
        "Token-generation bottleneck: don't generate wasted tokens.",
        prior="Lowers latency on prompts that finish early."),

    Experiment(27, "Persistent engine reuse (no reload)", "system", "best",
        {"warmup_runs": 2, "label": "warm_persistent"},
        "Keeping the model resident + extra warmup yields steady-state tok/s.",
        "Model-loading/warmup overhead removed from the hot path.",
        prior="Improves measured steady-state; standard serving practice."),

    Experiment(28, "Pinned host memory", "memory", "best",
        {"pin_memory": True, "label": "pinned_mem"},
        "Pinned memory speeds host<->device transfers.",
        "Targets H2D copy cost in the input pipeline.",
        prior="Mainly a CUDA win; near-neutral on unified-memory MPS.",
        expect_unsupported_on="mps"),

    Experiment(29, "Static cache + max-autotune compile", "runtime", "best",
        {"cache_implementation": "static", "torch_compile": True,
         "compile_mode": "max-autotune", "label": "static+autotune"},
        "Static cache gives compile a fixed-shape graph -> best fusion.",
        "Combine two prior wins; classic fast-decode recipe.",
        prior="Expected strong combo if both individually helped."),

    Experiment(30, "SDPA + fp16 combo", "runtime", "best",
        {"attn_implementation": "sdpa", "dtype": "float16", "label": "sdpa+fp16"},
        "Fused attention on half precision compounds two wins.",
        "Attack compute and bandwidth bottlenecks together.",
        prior="Likely additive."),

    Experiment(31, "Lazy mmap weight loading (safetensors)", "ssd_streaming", "best",
        {"low_cpu_mem_usage": True, "label": "mmap_safetensors"},
        "Memory-map weights so pages load on demand from SSD.",
        "Dense-model analogue of Flash-MoE SSD streaming; cuts load RAM.",
        prior="Helps load/peak-RAM; Flash-MoE found mmap bad for COLD per-token "
              "expert access, but fine for resident dense weights."),

    Experiment(32, "Expert SSD streaming (MoE-only)", "ssd_streaming", "best",
        {"label": "moe_ssd_stream(NA_dense)"},
        "Stream only the K active experts per layer from SSD on demand.",
        "Core MoE-on-laptop technique; the 0.5B model is DENSE so this is N/A "
        "here and is validated in Phase 2 on Qwen3.5-397B-A17B.",
        prior="N/A on dense 0.5B; documented and deferred to large-model phase.",),

    Experiment(33, "fp16 KV cache, fp32 compute", "memory", "best",
        {"kv_cache_dtype": "float16", "cache_implementation": "static", "label": "fp16_kv_cache"},
        "Store K/V in fp16 to halve cache memory while computing in fp32.",
        "KV-cache memory bottleneck at long context on 24GB.",
        prior="Memory win; small/no speed change for short prompts."),

    Experiment(34, "Attention slicing", "memory", "best",
        {"attention_slicing": True, "label": "attn_slicing"},
        "Compute attention in chunks to cap peak memory.",
        "Lets longer contexts fit under the 24GB ceiling.",
        prior="Trades a little speed for headroom.",),

    Experiment(35, "Length-adaptive max tokens", "decoding", "best",
        {"max_tokens_scale": 0.75, "label": "adaptive_maxtok"},
        "Cap generation per category to avoid runaway decode.",
        "Token-generation bottleneck; stop spending on low-value tokens.",
        prior="Latency win; must not truncate needed output (quality check).",
        expect_quality_risk=True),

    Experiment(36, "float32 matmul precision = high (TF32-style)", "runtime", "best",
        {"matmul_precision": "high", "label": "matmul_high"},
        "Allow reduced-precision fp32 matmul for throughput.",
        "Compute bottleneck on the GPU matmuls.",
        prior="Backend-dependent; measure for MPS."),

    Experiment(37, "Compile + SDPA + bf16 stack", "runtime", "best",
        {"torch_compile": True, "compile_mode": "reduce-overhead",
         "attn_implementation": "sdpa", "dtype": "bfloat16", "label": "compile+sdpa+bf16"},
        "Stack the three most promising runtime/precision wins.",
        "Compound optimization once individual winners are known.",
        prior="Either the new best or reveals a bad interaction."),

    Experiment(38, "Cached GenerationConfig reuse", "runtime", "best",
        {"reuse_generation_config": True, "label": "gen_config_cache"},
        "Reuse a prebuilt GenerationConfig to skip per-call validation.",
        "Per-call Python overhead in generate().",
        prior="Tiny win; cheap to keep if non-negative."),

    Experiment(39, "All cores vs P-cores (threads)", "system", "best",
        {"num_threads": 10, "label": "cpu_10threads_all"},
        "Use all 10 CPU cores for CPU-side ops.",
        "System scheduling: compare with exp18's P-core pinning.",
        prior="E-cores may add contention; compare to exp18.",),

    Experiment(40, "Batch tokenizer encode", "runtime", "best",
        {"batch_tokenize": True, "label": "batch_tokenize"},
        "Encode all prompts in one tokenizer call.",
        "Tokenizer overhead at the suite level.",
        prior="Helps suite wall-time, not per-token decode."),

    Experiment(41, "Dynamic batching emulation", "scheduling", "best",
        {"batch_size": 4, "label": "dynamic_batch"},
        "Group concurrently-arriving prompts into one forward pass.",
        "Continuous-batching-style scheduling for serving throughput.",
        prior="Throughput win under concurrent load."),

    Experiment(42, "Final stacked best-of-all", "runtime", "best",
        {"label": "final_stacked"},
        "Apply every validated win together as the optimized engine candidate.",
        "Produce the final optimized configuration.",
        prior="Should equal or beat every individual experiment."),
]


def get(exp: int) -> Experiment:
    for e in REGISTRY:
        if e.exp == exp:
            return e
    raise KeyError(exp)


def categories() -> List[str]:
    return sorted({e.category for e in REGISTRY})
