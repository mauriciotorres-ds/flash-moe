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

