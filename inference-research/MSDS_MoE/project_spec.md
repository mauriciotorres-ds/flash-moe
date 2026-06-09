# AI-Assisted Inference Engine Research Project

## Project Objective

You are an AI Systems Researcher and Inference Engineer.
Your mission is to design, benchmark, optimize, and analyze a **local Mixture-of-Experts (MoE) inference engine** that runs a large MoE model on a consumer laptop using an autoresearch methodology inspired by the Flash-MoE project.

The goal is not to train a new model.
The core idea is to exploit the defining property of MoE models — **only a small subset of experts (K per layer) is active for each token** — so that most expert weights can live on local SSD / memory-mapped storage and only the selected experts are loaded during decoding. This is what makes "large model on small machine" inference possible.

The goal is to:

1. Build a reproducible baseline MoE inference engine.
2. Run at least 40 benchmarked optimization experiments on a single real MoE model.
3. Discover which optimizations actually improve performance.
4. Document both successful and failed experiments.
5. Build an optimized inference engine whose central technique is on-demand expert streaming.
6. Build a real-time observability dashboard demonstrating the differences between baseline and optimized inference.
7. Produce a short report explaining *why* MoE sparsity makes large-model laptop inference possible.

The final result should resemble an inference-engine research project rather than a standard software engineering project.

---

## Target Models

This project runs on **real Qwen MoE models across three size tiers**, so the engine can be developed quickly on the smallest and then shown to scale to much larger models on the *same* laptop. Every tier is a genuine sparse MoE — only K experts activate per token — so the expert-streaming work is real at every size. (Do **not** substitute a small *dense* model: with no experts there is nothing to stream and the project loses its point.)

| Tier | Model | Total / Active | Q4_K_M GGUF | Role |
|---|---|---|---|---|
| **Small** | Qwen1.5-MoE-A2.7B | 14.3B / 2.7B | ~9–10 GB | Primary development + fast 40-experiment loop |
| **Medium** | Qwen3.5-35B-A3B | 35B / 3B | ~20–22 GB | Main modern-quality benchmark |
| **Large** *(optional)* | Qwen3-235B-A22B | 235B / 22B | ~142 GB | Scaling stretch — "235B on a 32 GB laptop"; download only after experiments, skip if memory/disk is insufficient |

All three fit on disk together (~172 GB; see the download section).

**The experiment cycle is run on the Small model only.** All infrastructure (baseline engine, benchmark harness, metrics, experiment framework, logging, dashboard) is built and validated on the **Small** tier, and the **entire 40+ experiment optimize/benchmark/keep-or-discard cycle happens exclusively on Qwen1.5-MoE-A2.7B**.

**The Medium and Large models are never experimented on or tuned.** They exist only so the *finished* inference engine — with the optimizations already selected on Small — can be pointed at them and **run** (a baseline pass and an optimized pass each) to demonstrate that the same engine and its validated optimizations generalize to larger MoEs. No experiment numbers, no keep/discard decisions, and no `expNNN.md` files are produced for Medium or Large; their runs are recorded only as benchmark/demo results (throughput, page-cache hit rate, memory) for the report and dashboard.

**Large (235B) is optional.** Download it only *after* the Small experiment cycle is complete and the engine is validated on Medium, and only if there is enough free disk (~142 GB) and it can realistically run on the available RAM. **If memory or disk is insufficient, skip the Large tier** — the project is complete with **Small (2.7B)** and **Medium (35B)** alone. Large is a bonus scaling demonstration, never a requirement.

### Hardware reality (32 GB-RAM reference laptop)
Active params per token — not total size — governs throughput, because only the active experts stream from SSD each token:
- **Small / Medium (2.7–3B active):** the active set stays warm in the OS page cache → real-time throughput (the instructor reached ~16 TPS on 32 GB).
- **Large (22B active):** ~10–12 GB streams per token, too large to stay warm in 32 GB → expect **<1 TPS**, SSD-bandwidth-bound. This is a legitimate, documented finding that shows *where* laptop streaming breaks down — not a failure. The cross-tier comparison is itself a headline result.

### Reference Data Point

A reference result from the instructor's own experiment: an inference engine was built from scratch for **Qwen3.5-35B-A3B on a MacBook Pro M1 Pro with 32 GB RAM**, including GGUF parsing, on-demand expert streaming, CPU/NEON optimizations, selected Metal kernels, and mmap/mlock-based expert weight access. Through ~40 systematic experiments, decoding throughput improved from **below 1 TPS to about 16.3 TPS**, exceeding a 15 TPS target, in two days. Your throughput target is a function of your own hardware, but the methodology — iterative profiling and targeted optimization of the expert-streaming hot path — is the same.

---

## Downloading the Model Weights

For each tier, use the **GGUF 4-bit (`Q4_K_M`) quantization**. GGUF is the format the engine streams from: weights are memory-mapped and individual experts can be read on demand. Do **not** use the full bf16 safetensors releases — they are unquantized and intended for vLLM/Transformers, not SSD expert streaming. All three quants together are ~172 GB; keep them **outside the git repo** so they are never committed.

| Tier | GGUF repo (primary) | Alternates | ~Size |
|---|---|---|---|
| Small | `RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf` | `PrunaAI/Qwen1.5-MoE-A2.7B-Chat-GGUF-smashed` | ~9.5 GB |
| Medium | `unsloth/Qwen3.5-35B-A3B-GGUF` | `Qwen/Qwen3.5-35B-A3B-GGUF`, `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` | ~20–22 GB |
| Large | `unsloth/Qwen3-235B-A22B-GGUF` | `Qwen/Qwen3-235B-A22B-GGUF`, `bartowski/Qwen_Qwen3-235B-A22B-GGUF` | ~142 GB |

### One-time setup

```powershell
pip install -U "huggingface_hub[hf_transfer]"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"   # faster parallel downloads
```

### Inspect a repo before downloading

GGUF filenames and shard counts vary by publisher (the Large tier is split into many shards). List the files first to find the exact `Q4_K_M` name:

```powershell
hf download <repo-id> --repo-type model --dry-run
# or browse the repo's /tree/main page on huggingface.co
```

### Download the Q4_K_M quant for each tier

Download just the one quant per tier by glob pattern into its own local folder (not the whole repo, which contains every quant level). The pattern also captures all shards if the quant is split:

```powershell
# Small — primary development model (download this first)
hf download RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf `
  --include "*Q4_K_M*.gguf" `
  --local-dir "C:\Users\garre\school\summer_2026\ml_cyber\models\Qwen1.5-MoE-A2.7B-GGUF"

# Medium
hf download unsloth/Qwen3.5-35B-A3B-GGUF `
  --include "*Q4_K_M*.gguf" `
  --local-dir "C:\Users\garre\school\summer_2026\ml_cyber\models\Qwen3.5-35B-A3B-GGUF"

# Large (~142 GB, sharded — download last)
hf download unsloth/Qwen3-235B-A22B-GGUF `
  --include "*Q4_K_M*.gguf" `
  --local-dir "C:\Users\garre\school\summer_2026\ml_cyber\models\Qwen3-235B-A22B-GGUF"
```

If a quant is sharded (e.g. `...Q4_K_M-00001-of-0000N.gguf`), point the engine at the **first** shard; GGUF loaders resolve the rest automatically. The `--include` glob is case-sensitive, so match the casing the repo actually uses (`q4_k_m` vs `Q4_K_M`) — check with `--dry-run` first.

### Notes
- **Auth:** the GGUF mirrors are normally ungated. If a download returns HTTP 401/403, run `hf auth login` with a Hugging Face token and accept the model license on its page first.
- **llama.cpp compatibility:** the Qwen3/3.5 MoEs use newer (and hybrid Gated-DeltaNet) architectures. Confirm each GGUF repo's README states current-`llama.cpp` support before relying on a third-party runtime; the older Qwen1.5-MoE (Qwen2MoE arch) is broadly supported, but the larger Qwen3 MoEs may need a recent build.
- **Verification:** after each download, record the file size and SHA/commit hash in `state.json` (or a `MODEL.md`) so the exact weights used are reproducible.
- **Order:** download the **Small** tier first and build everything against it; only pull **Medium** once the 40-experiment cycle is working.
- **Large tier is optional and download-last:** do **not** download the **Large** model (Qwen3-235B-A22B, ~142 GB) until the full experiment cycle on Small is complete *and* the engine has been validated on Medium. Before downloading it, confirm there is enough free disk (~142 GB plus headroom) **and** that it can realistically run — with only ~22B active params/token it streams ~10–12 GB per token, which a 32 GB-RAM machine cannot keep warm. **If there is not enough memory or disk, skip the Large tier entirely** and treat the project as complete with the **Small (2.7B)** and **Medium (35B)** models only. The Large run is a bonus "scaling" demonstration, never a requirement.

---

## Session Continuity and Progress Persistence

**This project is designed to survive interruptions.**

Context windows end. Internet connections drop. Chat sessions are killed. The project must continue correctly after any interruption without losing work or requiring experiments to be re-run.

### Persistence Rules

**After every experiment:**
- Write results immediately to `results/experiments.csv` and `results/experiments.json`.
- Write the experiment markdown file (`experiments/expNNN.md`) before moving to the next experiment.
- Commit all changes to git with a descriptive message (e.g., `exp012: mmap expert streaming +8% tok/s — keep`).

**State file:**
Maintain a `state.json` file in the project root at all times. Update it after every experiment.

```json
{
  "current_experiment": 13,
  "last_completed_experiment": 12,
  "baseline_tps": 1.2,
  "best_tps": 9.4,
  "best_config": "exp009",
  "status": "running",
  "last_updated": "2026-06-04T14:32:00"
}
```

**Session resume protocol:**
At the start of every new session (or after any interruption), read `state.json` first. Do not re-run completed experiments. Resume from `current_experiment`. If `status` is `"running"`, the previous session was interrupted mid-experiment — discard partial results and re-run that experiment cleanly.

**Checkpoint at milestones:**
- After the baseline engine works: commit + tag `baseline-complete`
- After the optimization cycle (40+ experiments): commit + tag `optimization-complete`
- After the dashboard is complete: commit + tag `dashboard-complete`

### What Must Always Be On Disk

| File | Updated When |
|---|---|
| `state.json` | After every experiment |
| `results/experiments.csv` | After every experiment |
| `results/experiments.json` | After every experiment |
| `results/leaderboard.csv` | After every keep decision |
| `results/benchmark_history.csv` | After every benchmark run |
| `results/best_config.json` | When best is updated |
| `experiments/expNNN.md` | Before starting exp N+1 |
| `failure_log.md` | After every discard decision |

No experiment is considered complete until its results exist on disk.

---

## Baseline Engine Requirements

Create a reproducible **baseline MoE inference engine** that runs the Small model (Qwen1.5-MoE-A2.7B) end to end. The baseline is the naive, unoptimized reference: it loads the model the simplest way that works (e.g., full GGUF load, or all weights resident / faulted in with no streaming strategy) and decodes greedily. Its only job is to be a correct, reproducible control that every optimization is measured against.

Measure:
- Tokens generated
- Tokens per second
- Time to first token
- Total latency
- Peak memory usage (resident set size)
- CPU utilization
- GPU utilization (if a GPU path is used)
- Context length
- **Expert load volume per token** (bytes read for active experts) and **page-cache hit rate** for expert reads

Store all baseline benchmark results.

---

## Required Per-Generation Telemetry

Every generation (baseline or optimized, from the benchmark harness or the dashboard) must record and persist the following as first-class fields, not just aggregate summaries:

- **`total_runtime_s`** — end-to-end wall-clock time for the generation (prompt encode → final token). The benchmark harness must also record a **suite-level total runtime** across all prompts in a run.
- **`tokens_per_second`** — tokens generated ÷ `total_runtime_s` (also keep the rolling/instantaneous tok/s used for live display).
- **`expert_selection`** — the router's chosen experts, recorded **per decoding step, per MoE layer**: the top-K expert indices (and, where available, their routing weights) selected for each token. This is the direct evidence of MoE sparsity in action. Keep the existing aggregate views (activation frequency, tokens routed per expert, heatmap) as roll-ups of this per-step record.

These three must be written to disk with each run (e.g., in `results/` and/or `dashboard_logs/`) so they are reproducible and inspectable after the fact, and they must be surfaced on the dashboard (see below). The per-step `expert_selection` log may be large; it is acceptable to store it as a compact per-run artifact (e.g., a `.jsonl` or array keyed by `step → layer → [expert_ids]`).

---

## Benchmark Harness

Create: `benchmark_runner.py`

The benchmark suite should include:

**Factual Prompts** — Knowledge and QA tasks.

**Coding Prompts** — Code generation and debugging tasks.

**Long Reasoning Prompts** — Multi-step reasoning tasks.

**Summarization Prompts** — Document summarization tasks.

**Structured Output Prompts** — JSON, YAML, and table generation.

All benchmark runs must be reproducible (fixed seed, fixed prompts, fixed token budgets).

---

## AI Autoresearch Methodology

Follow an autoresearch workflow inspired by Flash-MoE.

The objective is not to perform 40 random experiments. The objective is to discover through measurement, profiling, and iteration which inference optimizations actually matter — especially those on the **expert-streaming hot path**, which is the dominant cost for MoE-on-laptop decoding.

For every experiment:
1. Analyze benchmark results.
2. Identify the largest bottleneck.
3. Form a hypothesis.
4. Implement a targeted optimization.
5. Benchmark.
6. Compare against current best.
7. Document findings.
8. **Write results to disk immediately.**
9. Use findings to guide the next experiment.

Experiments should build on previous discoveries.

---

## Bottleneck-Driven Development

Before every experiment, profile the system. Identify:
- Expert I/O bottlenecks (cold vs. warm expert reads, SSD bandwidth, page-cache pressure)
- Compute bottlenecks (dequant, matvec, attention)
- Memory bottlenecks (resident set, mmap behavior)
- KV cache bottlenecks
- Token generation bottlenecks
- Scheduling bottlenecks
- Model loading bottlenecks

Prioritize experiments that attack the largest bottleneck. For this model the expert-streaming I/O path is expected to dominate early on.

---

## Trust Measurements Over Intuition

Never assume an optimization helps. Every optimization must be validated with benchmark results. Measured data takes priority over intuition.

Never fabricate:
- benchmark results
- profiling results
- experiment outcomes
- speedups
- latency improvements

All reported metrics must come from actual execution on the real model.

---

## Optimization Categories

**SSD / MoE Expert Streaming** *(core technique — the project centers on this)*
- Store expert weights on SSD or memory-mapped storage; keep only non-expert weights resident.
- At each token, read the router's top-K expert indices and load **only those experts** for each layer on demand.
- Benchmark cold vs. warm expert reads.
- Measure page-cache hit rate vs. throughput tradeoffs ("trust the OS" vs. custom caches).
- Compare access strategies: `pread`, `mmap`, `mmap` + `mlock`, async/prefetch I/O.
- Explore expert caching policies (LRU, keep-shared-expert-resident) and expert prefetch/prediction.

**Runtime Optimizations**
- `torch.inference_mode()` / `torch.no_grad()`
- `torch.compile()` / graph capture
- optimized generation loops
- tokenizer optimizations
- GGUF parsing / fast model load

**Memory Optimizations**
- KV cache improvements and cache layouts
- memory reuse
- pinned memory
- memory mapping of weights

**Quantization**
- FP16 / BF16
- INT8
- 4-bit (GGUF Q4 and lower) expert weights
- mixed precision

**Compute Kernels** *(platform-dependent, optional)*
- fused dequant + matvec for quantized experts
- CPU SIMD (AVX2/AVX-512/NEON) inner loops
- GPU kernels (Metal/CUDA) where available

**Scheduling**
- prompt batching
- dynamic batching
- continuous batching
- request scheduling

**Decoding**
- speculative decoding / draft models
- alternative generation strategies
- early exit / early EOS

**System Optimizations**
- multiprocessing / threading / async execution
- model loading improvements

A credible project will include a substantial number of experiments in the **Expert Streaming** category — it is the technique the assignment is grading.

---

## Minimum Experiment Requirement

Run at least **40 unique benchmarked experiments**, all on the **Small** model (Qwen1.5-MoE-A2.7B). The Medium and Large models are only *run* by the finished engine and never count toward the experiment requirement.

Every experiment must contain:

```
Experiment Number:
Title:
Hypothesis:
Implementation:
Files Modified:
Benchmark Results:
Performance Change:
Decision:
Lessons Learned:
```

---

## Failure Documentation

Failed experiments are valuable. Do not hide failures.

For every failed experiment document:
- Why it was attempted
- Why it failed
- Performance impact
- Lessons learned

Maintain: `failure_log.md`

---

## Experiment Retention Policy

Keep changes only if they:
- Improve throughput
- Reduce latency
- Reduce memory usage
- Preserve output quality

Otherwise revert the change. The final optimized engine should contain only validated improvements.

---

## Required Outputs

```
results/
  experiments.csv
  experiments.json
  leaderboard.csv
  benchmark_history.csv
  best_config.json

experiments/
  exp001.md
  exp002.md
  ...
  exp040.md

reports/
  final_report.md

state.json          ← session continuity checkpoint
failure_log.md
baseline_engine.py
optimized_engine.py
```

---

## What We Tried (And What Worked)

Generate a dedicated report section containing:
- Top 10 successful optimizations
- Top 10 failed optimizations
- Largest speedup
- Largest latency reduction
- Largest memory reduction
- Most surprising finding
- Highest ROI optimization
- Optimization that looked promising but failed
- Optimization that improved one metric while harming another

---

## Final Engine Selection

After all experiments, create:
- `baseline_engine.py`
- `optimized_engine.py`

The optimized engine must be the highest-performing validated configuration, and its expert-streaming strategy must be the best one discovered through the experiments.

---

## Final Report

Produce `reports/final_report.md` (a short written report) that includes:
- The optimization progression (TPS across experiments, baseline → best).
- The "What We Tried (And What Worked)" section above.
- **A clear explanation of why MoE sparsity makes "large model on small machine" inference possible** — i.e., why activating only K experts per token means most weights can stay on SSD and only a few megabytes need to be read per layer per token.

---

## Interactive Inference Dashboard

**IMPORTANT:** The dashboard must be built AFTER the optimization cycle is complete.

Launch command:
```bash
streamlit run dashboard.py
```

The dashboard should function as a professional inference-engine observability and research platform.

### Live Inference Playground

Users must be able to:
- Enter prompts
- Select model (Small / Medium / Large tier)
- Select engine (Baseline / Optimized / Compare Both)
- Configure generation parameters
- Run inference directly

Controls: prompt textbox, system prompt textbox, max tokens, temperature, top-p, top-k, seed.

Model Selector: Qwen1.5-MoE-A2.7B (Small) / Qwen3.5-35B-A3B (Medium) / Qwen3-235B-A22B (Large)
Engine Selector: Baseline Engine / Optimized Engine / Compare Both

### Real-Time Streaming Generation

Generated text must stream live. Display: generated text, tokens generated, elapsed time, current tok/s, time to first token.

### Real-Time Performance Monitoring

While generation is executing display: tok/s, rolling tok/s, **total runtime (live elapsed wall-clock, finalized as `total_runtime_s`)**, latency, time to first token, peak memory, current memory, CPU utilization, GPU utilization, KV cache size, context length, active engine, and **expert-streaming stats** (bytes read per token, page-cache hit rate).

### Live Expert Selection

While generation is executing, display **which experts are chosen** in real time — the per-step, per-layer top-K expert indices (and routing weights where available) from the `expert_selection` telemetry. Show, at minimum:
- the experts activated for the most recent token (per MoE layer), and
- a running tally / heatmap that updates as tokens stream.

This is the dashboard's direct demonstration of MoE sparsity: the user can watch only K experts light up per token while the rest stay on SSD. Total runtime and tokens/second must be shown prominently alongside it.

### Baseline vs Optimized Comparison Mode

Run baseline only / optimized only / both simultaneously. Display side-by-side output and metrics. Calculate `speedup = optimized_tps / baseline_tps`.

### Live Visualization

Continuously updating charts: tok/s over time, memory over time, CPU usage over time, GPU usage over time, expert-read volume over time.

### Experiment Explorer

Browse, sort, and filter experiments. View accepted and rejected optimizations. For each experiment: hypothesis, category, results, decision, lessons learned.

### Optimization Timeline

Charts: Experiment Number vs tok/s, latency, memory, speedup.

### Optimization Diff Viewer

Show baseline vs. optimized configuration. Display runtime, quantization, cache, scheduling, expert-streaming, and system-level changes.

### MoE Visualization

For the selected MoE model (any tier), display expert utilization: activation frequencies, load distribution, tokens routed per expert, and expert heatmaps. This visualization is central — it makes the sparsity the project exploits visible, and comparing it across tiers shows how the active-expert fraction changes with model size. If a metric is unavailable from the engine, document the limitation.

### Session Logging

Every dashboard run should optionally record: prompt, engine, output, metrics, timestamp. Store in `dashboard_logs/`.

---

## Final User Experience Goal

A user should be able to:
1. Open the dashboard.
2. Enter a prompt.
3. Select baseline or optimized inference.
4. Watch generation stream live.
5. Observe tok/s changing in real time.
6. Observe memory, hardware utilization, and expert-streaming I/O.
7. Compare baseline and optimized performance.
8. Explore the optimization history.
9. See which experts activate for their prompt.
10. Understand exactly how the optimized engine was created — and why MoE sparsity makes it possible.
