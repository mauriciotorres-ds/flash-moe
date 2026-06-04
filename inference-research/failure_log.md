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

