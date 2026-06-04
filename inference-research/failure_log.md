# Failure Log

Discarded experiments and why. Failures are data.

## exp001 — torch.no_grad() context

- **Why attempted:** Disabling autograd bookkeeping reduces per-step overhead.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -3.22% vs best
- **Lessons learned:** Small but free win on most backends.

## exp002 — torch.inference_mode() context

- **Why attempted:** inference_mode is stricter than no_grad and skips version counters.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -0.94% vs best
- **Lessons learned:** Usually >= no_grad; expected to replace it.

## exp003 — model.eval() isolation

- **Why attempted:** Leaving train mode on enables dropout/other train-time paths.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -5.47% vs best
- **Lessons learned:** eval() expected better; this run should be slower/worse quality.

## exp004 — float16 weights

- **Why attempted:** Half precision halves memory traffic; MPS has fast fp16 paths.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -8.12% vs best
- **Lessons learned:** Often the single biggest win on Apple GPUs.

## exp005 — bfloat16 weights

- **Why attempted:** bf16 keeps fp32 dynamic range with fp16 bandwidth.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -9.47% vs best
- **Lessons learned:** Throughput similar to fp16; possibly better numerics.

## exp007 — FlashAttention-2

- **Why attempted:** FA2 reduces memory movement in attention.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. load fallback (dropped optional kwargs): FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package for FlashAttention2 doesn't seem to be installed.
- **Performance impact:** -1.56% vs best
- **Lessons learned:** Expected UNSUPPORTED on MPS -> graceful fallback (a failure to log).

## exp008 — SDPA math backend

- **Why attempted:** Forcing the math kernel can be faster for tiny head dims.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -2.02% vs best
- **Lessons learned:** Backend-dependent; measure.

## exp009 — SDPA mem-efficient backend

- **Why attempted:** Memory-efficient attention lowers peak memory.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -2.55% vs best
- **Lessons learned:** May trade a little speed for memory.

## exp010 — use_cache ablation

- **Why attempted:** Disabling the KV cache forces full recompute each step.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -54.19% vs best
- **Lessons learned:** Expected MUCH slower; validates cache value.

## exp012 — Offloaded KV cache

- **Why attempted:** Offloading cache to CPU frees GPU memory for weights.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.; cache_implementation='offloaded' requires CUDA; not supported on MPS. Running with dynamic cache.
- **Performance impact:** -12.55% vs best
- **Lessons learned:** Likely slower for short prompts; helps long-context only.

## exp013 — torch.compile (default)

- **Why attempted:** Graph capture fuses ops and cuts dispatch overhead.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -13.65% vs best
- **Lessons learned:** First-call compile cost; warmup hides it.

## exp014 — torch.compile (reduce-overhead)

- **Why attempted:** CUDA-graph-style overhead reduction for the decode loop.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -15.16% vs best
- **Lessons learned:** Often best compile mode for autoregressive decode.

## exp015 — torch.compile (max-autotune)

- **Why attempted:** Autotuned kernels maximise throughput.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -16.74% vs best
- **Lessons learned:** Long compile; may or may not beat reduce-overhead.

## exp016 — low_cpu_mem_usage loading

- **Why attempted:** Streamed weight loading lowers peak RAM during init.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -18.08% vs best
- **Lessons learned:** Affects load time/peak RAM, not steady tok/s.

## exp017 — channels_last memory format

- **Why attempted:** channels_last can improve some kernel memory access.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -17.62% vs best
- **Lessons learned:** Usually neutral for transformers; measure.

## exp018 — Pin threads to performance cores

- **Why attempted:** On CPU paths, limiting to the 4 P-cores avoids E-core contention.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -19.35% vs best
- **Lessons learned:** Only relevant if a CPU fallback path is used.

## exp019 — Slow (Python) tokenizer ablation

- **Why attempted:** The Rust fast tokenizer should beat the Python one at encode.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -33.18% vs best
- **Lessons learned:** Fast tokenizer expected better; this run should regress.

## exp020 — INT8 (bitsandbytes)

- **Why attempted:** 8-bit weights cut memory traffic ~4x vs fp32.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. quantization=int8 via bitsandbytes requires CUDA; not supported on mps. Running unquantized.
- **Performance impact:** -35.07% vs best
- **Lessons learned:** bitsandbytes is CUDA-only -> expected UNSUPPORTED on MPS.

## exp021 — NF4 4-bit (bitsandbytes)

- **Why attempted:** 4-bit weights for maximum memory reduction.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. quantization=nf4 via bitsandbytes requires CUDA; not supported on mps. Running unquantized.
- **Performance impact:** -32.60% vs best
- **Lessons learned:** CUDA-only; expected UNSUPPORTED on MPS. Quality risk if forced.

## exp022 — Prompt batching (bs=4)

- **Why attempted:** Batching amortises kernel launches across prompts.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -33.23% vs best
- **Lessons learned:** Improves throughput, not single-stream latency.

## exp023 — Prompt batching (bs=8)

- **Why attempted:** Larger batch -> higher utilisation until memory-bound.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -36.96% vs best
- **Lessons learned:** Diminishing returns / memory pressure past some point.

## exp024 — Speculative decoding (draft model)

- **Why attempted:** A small draft proposes tokens the target verifies in parallel.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -85.20% vs best
- **Lessons learned:** Gains depend on draft acceptance rate; can break even.

## exp025 — Greedy vs sampling cost

- **Why attempted:** Greedy avoids sampling/softmax-top-k overhead.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -16.23% vs best
- **Lessons learned:** Greedy slightly faster; default decode for benchmarking.

## exp026 — Early EOS stopping

- **Why attempted:** Stop as soon as EOS is produced instead of padding to max.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -17.10% vs best
- **Lessons learned:** Lowers latency on prompts that finish early.

## exp027 — Persistent engine reuse (no reload)

- **Why attempted:** Keeping the model resident + extra warmup yields steady-state tok/s.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -19.17% vs best
- **Lessons learned:** Improves measured steady-state; standard serving practice.

## exp028 — Pinned host memory

- **Why attempted:** Pinned memory speeds host<->device transfers.
- **Why it failed / was discarded:** Did not beat current best by >= 1% or degraded quality. 
- **Performance impact:** -15.84% vs best
- **Lessons learned:** Mainly a CUDA win; near-neutral on unified-memory MPS.

