# The Inference Engine — How It Works

This document explains the engine layer on its own: how baseline, configurable,
and optimized engines relate, what every optimization knob does, and how a
single code path keeps all 42 experiments fair. For the dashboard see
[DASHBOARD.md](DASHBOARD.md); for the charts see [PLOTS.md](PLOTS.md).

**Hardware target:** Apple M4 · 24 GB unified memory (MPS backend). The code
auto-detects the host, so it also runs on CUDA or CPU, but every default and
every plot label assumes the M4.

## One engine, driven by config

There is exactly **one** engine implementation, `ConfigurableEngine`
(`ireng/engine.py`). Everything else is a configuration of it:

- `baseline_engine.py` hands it the frozen **baseline** config: plain Hugging
  Face Transformers, float32, eager attention, dynamic KV cache, no tricks.
  This is the control that every experiment is measured against.
- `optimized_engine.py` hands it the **best** config — read from
  `results/best_config.json`, which the autoresearch runner writes whenever a
  new best is validated. Until you have run the real experiments, it falls back
  to a documented default (inference_mode + fp16 + SDPA + static cache + greedy).
- Each **experiment** is a small dict of overrides applied to either the
  current-best config (the autoresearch default — experiments build on prior
  wins) or the clean baseline (for ablations that isolate one knob).

This is the central design decision. Because only the config changes between
runs, a difference in tokens/second is attributable to the knob under test and
nothing else. It is also why the optimized engine is not a separate codebase to
maintain — it is just a saved configuration.

```
EngineConfig ──▶ ConfigurableEngine.load() ──▶ .stream(prompt) ──▶ tokens + GenerationMetrics
     ▲                                                                     │
 baseline / best / experiment-delta                          benchmark.py aggregates
```

## The lifecycle of a generation

1. **Load** (`ConfigurableEngine.load`): resolve device, set thread count and
   matmul precision, build the tokenizer, apply any quantization config, load
   the model with the chosen dtype and attention implementation, move it to the
   device, optionally `model.eval()`, `torch.compile`, attach a speculative
   draft model. torch/transformers are imported lazily so the framework can be
   inspected without a GPU or the model present.
2. **Format**: the prompt's system+user messages go through the model's chat
   template (`apply_chat_template`), falling back to a plain concatenation.
3. **Stream** (`ConfigurableEngine.stream`): generation runs in a background
   thread feeding a `TextIteratorStreamer`. The first emitted chunk marks
   **time-to-first-token**; each subsequent chunk updates live tok/s. A
   background `ResourceSampler` records CPU%, RSS memory, and (where available)
   GPU%.
4. **Finalize**: compute total latency, decode-only tok/s (excluding TTFT),
   peak/current memory, and an architecture-based **KV-cache size** estimate.

## What every knob does

The knobs live in `EngineConfig` (`ireng/config.py`). Grouped by the spec's
optimization categories:

**Runtime** — `inference_mode` / `no_grad` (skip autograd bookkeeping),
`eval_mode`, `torch_compile` + `compile_mode` (graph capture / kernel fusion),
`matmul_precision` (reduced-precision fp32 matmul), `num_threads`,
`channels_last`, `reuse_generation_config`, `batch_tokenize`.

**Attention / cache** — `attn_implementation` (eager / sdpa /
flash_attention_2), `sdpa_backend` (math / flash / mem_efficient), `use_cache`,
`cache_implementation` (dynamic / static / offloaded), `kv_cache_dtype`,
`attention_slicing`.

**Precision / quantization** — `dtype` (float32 / float16 / bfloat16),
`quantization` (none / int8 / nf4 / fp4) with `quant_backend`. **Note:**
bitsandbytes INT8/4-bit is CUDA-only; on Apple MPS the engine records that it is
unsupported and runs unquantized rather than crashing or faking a result. The
memory-bandwidth win on Apple Silicon therefore comes from fp16/bf16.

**Loading / memory** — `low_cpu_mem_usage` (streamed/mmap weight load —
the dense-model analogue of Flash-MoE's SSD streaming), `pin_memory`.

**Decoding** — `do_sample` + `temperature`/`top_p`/`top_k`/`seed`,
`speculative` + `assistant_model_id` (draft model), `early_stop_eos`,
`max_tokens_scale` (length-adaptive cap).

**Scheduling** — `batch_size` (>1 enables prompt/dynamic batching for
throughput).

Knobs the HF stack cannot express faithfully (e.g. a public per-layer KV dtype
switch, decoder attention slicing) are applied where the API allows and
otherwise recorded in `support_notes` with the honest reason — never silently
ignored.

## Graceful degradation, never fabrication

If a knob is unsupported on the host, the engine appends a note to
`support_notes` and continues with the nearest valid behaviour. The benchmark
records the real outcome (often a fallback that performs like the baseline). A
metric that cannot be obtained on a backend — GPU utilisation on MPS, which
Apple does not expose — is reported as `None` and shown as "n/a (MPS)". Nothing
is invented.

## Quality gating

A speedup that corrupts output is not a win (cf. Flash-MoE's 2-bit build that
broke JSON tool-calling). `ireng/benchmark.py` runs a cheap, deterministic
quality check per prompt — non-empty, not gibberish, valid JSON for the JSON
task, a pipe character for the table task — and the runner discards any config
that drops below the quality floor (0.7) even if it is faster.

## Running it

```bash
pip install -r requirements.txt          # torch, transformers, etc.

# single engines (downloads Qwen2.5-0.5B on first run)
python baseline_engine.py  --prompt "Explain MoE." --max-new-tokens 96
python optimized_engine.py --prompt "Explain MoE." --show-config

# full reproducible suite + speedup
python benchmark_runner.py --engine both --device mps
```

On the 397B MoE the Python HF path is not used; that model runs through the
Flash-MoE Metal engine in `../metal_infer`, driven by
`large_model/validate_transfer.py`.
