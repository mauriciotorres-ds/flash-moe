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

