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

## exp015 — n_gpu_layers=30 + flash_attn=True

- **Why attempted:** Partial offload + flash attn may outperform full offload if memory pressure limits full-GPU performance.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=99.1176 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp016 — n_gpu_layers=-1 + mlock=True

- **Why attempted:** Full GPU offload with locked expert pages reduces SSD reads during the decode loop.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=95.7766 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp017 — n_gpu_layers=-1 + no mmap (all RAM)

- **Why attempted:** All weights in RAM + all layers on GPU: maximum throughput if 24 GB can hold the model.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=95.3751 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp018 — n_threads=1 — single thread

- **Why attempted:** Single-thread baseline to measure threading overhead.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=20.9903 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp019 — n_threads=4

- **Why attempted:** 4 threads should improve CPU-side dequant and attention.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=65.7914 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp020 — n_threads=8

- **Why attempted:** 8 threads fully utilises performance cores on M4.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=87.4921 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp021 — n_threads=12

- **Why attempted:** 12 threads may exceed core count and add scheduling overhead.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=56.2615 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp022 — n_threads=12 (all logical CPUs)

- **Why attempted:** Using all logical CPUs saturates the CPU scheduler; may compete with the Metal command queue.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=57.5837 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp023 — flash_attn=True on CPU

- **Why attempted:** Flash attention reduces attention memory traffic even on CPU, potentially speeding up attention layers.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=57.6228 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp024 — flash_attn=True + best thread count

- **Why attempted:** Adding flash attn to the current best CPU config further reduces attention overhead.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=99.1319 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

## exp025 — Best GPU layers + best n_threads

- **Why attempted:** The best GPU offload and best thread count stack additively.
- **Why it failed / was discarded:** Did not improve tok/s above keep threshold.
- **Performance impact:** mean_tps=98.7972 vs best=98.9629
- **Lessons learned:** Config delta did not help on this hardware.

