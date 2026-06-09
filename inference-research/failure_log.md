# Failure Log

Discarded experiments from the MoE expert-streaming optimization cycle.
Failures are data — every discard teaches something.

Model: Qwen1.5-MoE-A2.7B Q4_K_M GGUF · Hardware: Apple M4 · 24 GB unified memory
## exp001 — No mmap — load all weights into RAM

- **Why attempted:** Disabling mmap forces eager load of all weights into RAM, eliminating page-fault overhead during expert access at the co
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=50.3131 vs best=51.2465
- **Lessons learned:** Config delta did not help on this hardware.

## exp002 — mmap + mlock — pin expert pages in RAM

- **Why attempted:** mlock prevents the OS from evicting expert pages, reducing cold-token SSD reads in long sessions.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=43.6297 vs best=51.2465
- **Lessons learned:** Config delta did not help on this hardware.

## exp003 — No mmap + mlock (all weights locked in RAM)

- **Why attempted:** Full eager load + locked pages eliminates all SSD I/O after load.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=50.2638 vs best=51.2465
- **Lessons learned:** Config delta did not help on this hardware.

## exp005 — n_batch=128 — smaller expert read batch

- **Why attempted:** Smaller batch reduces peak memory for prompt processing at the cost of more batching overhead.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=55.5806 vs best=58.2281
- **Lessons learned:** Config delta did not help on this hardware.

## exp006 — n_batch=1024 — larger expert read batch

- **Why attempted:** Larger batch amortises overhead for long prompts.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=58.7419 vs best=58.2281
- **Lessons learned:** Config delta did not help on this hardware.

## exp007 — n_batch=2048

- **Why attempted:** Maximum batch: best throughput for long prompts, highest memory.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=57.3486 vs best=58.2281
- **Lessons learned:** Config delta did not help on this hardware.

## exp009 — n_ctx=4096 — larger context window

- **Why attempted:** Larger context increases KV cache pressure and may slow expert streaming by reducing page cache headroom.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=40.1689 vs best=61.7317
- **Lessons learned:** Config delta did not help on this hardware.

## exp010 — n_ctx=1024 — moderate context

- **Why attempted:** 1024-token context balances KV memory and generation length.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=48.7842 vs best=61.7317
- **Lessons learned:** Config delta did not help on this hardware.

