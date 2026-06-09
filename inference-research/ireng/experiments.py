"""experiments.py — 40 MoE expert-streaming experiment definitions.

Per the spec, the experiment cycle runs on Qwen1.5-MoE-A2.7B (Small tier)
only.  The core technique is expert streaming from GGUF on SSD via mmap/pread.
Experiments target the expert-streaming hot path first, then GPU offload,
then threading, attention, and finally decoding/context trade-offs.

Each Experiment is metadata + config overrides.  The keep/discard decision
is made by the runner from real benchmark measurements, never predetermined.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_N_CPU = os.cpu_count() or 10


@dataclass(frozen=True)
class Experiment:
    exp:        int
    title:      str
    category:   str
    # "baseline" = always diff against exp000 baseline config
    # "best"     = diff against current best (autoresearch stacking)
    base:       str
    overrides:  dict[str, Any]
    hypothesis: str
    rationale:  str
    expect_quality_risk: bool = False


# ── Experiment registry ───────────────────────────────────────────────────────

EXPERIMENTS: list[Experiment] = [

    # ── 0: Baseline ────────────────────────────────────────────────────────
    Experiment(
        exp=0, title="Baseline (mmap=True, CPU-only, no flash-attn)",
        category="ssd_streaming", base="baseline",
        overrides={},
        hypothesis="Establish reproducible baseline tok/s on Qwen1.5-MoE-A2.7B GGUF.",
        rationale="exp000 is the reference all other experiments are measured against.",
    ),

    # ── Expert Streaming / SSD I-O (exp001–010) ────────────────────────────
    Experiment(
        exp=1, title="No mmap — load all weights into RAM",
        category="ssd_streaming", base="baseline",
        overrides={"use_mmap": False},
        hypothesis="Disabling mmap forces eager load of all weights into RAM, "
                   "eliminating page-fault overhead during expert access at the cost "
                   "of higher RSS.",
        rationale="Separates 'trust OS page cache' cost from 'no page faults' benefit.",
    ),
    Experiment(
        exp=2, title="mmap + mlock — pin expert pages in RAM",
        category="ssd_streaming", base="baseline",
        overrides={"use_mmap": True, "use_mlock": True},
        hypothesis="mlock prevents the OS from evicting expert pages, "
                   "reducing cold-token SSD reads in long sessions.",
        rationale="mlock is free when RAM is ample; harmful when RAM is scarce.",
    ),
    Experiment(
        exp=3, title="No mmap + mlock (all weights locked in RAM)",
        category="ssd_streaming", base="baseline",
        overrides={"use_mmap": False, "use_mlock": True},
        hypothesis="Full eager load + locked pages eliminates all SSD I/O after load.",
        rationale="Quantifies maximum possible throughput if 24 GB RAM can hold all weights.",
    ),
    Experiment(
        exp=4, title="Warm cache run (second inference, same process)",
        category="ssd_streaming", base="baseline",
        overrides={},   # same config — warm cache via repeated run
        hypothesis="OS page cache warms expert weights after the first token; "
                   "second run should show higher hit rate and faster tok/s.",
        rationale="Measures the page-cache benefit: cold vs warm throughput gap.",
    ),
    Experiment(
        exp=5, title="n_batch=128 — smaller expert read batch",
        category="ssd_streaming", base="baseline",
        overrides={"n_batch": 128},
        hypothesis="Smaller batch reduces peak memory for prompt processing "
                   "at the cost of more batching overhead.",
        rationale="Batch size controls prompt tokenization throughput.",
    ),
    Experiment(
        exp=6, title="n_batch=1024 — larger expert read batch",
        category="ssd_streaming", base="baseline",
        overrides={"n_batch": 1024},
        hypothesis="Larger batch amortises overhead for long prompts.",
        rationale="Batch size controls prompt throughput; bigger helps long inputs.",
    ),
    Experiment(
        exp=7, title="n_batch=2048",
        category="ssd_streaming", base="baseline",
        overrides={"n_batch": 2048},
        hypothesis="Maximum batch: best throughput for long prompts, highest memory.",
        rationale="Upper bound on batch size benefit.",
    ),
    Experiment(
        exp=8, title="n_ctx=512 — minimal context, small KV cache",
        category="ssd_streaming", base="baseline",
        overrides={"n_ctx": 512},
        hypothesis="Smaller context window reduces KV cache allocation, "
                   "freeing memory for more OS page cache for expert weights.",
        rationale="KV cache competes with expert page cache for unified memory.",
    ),
    Experiment(
        exp=9, title="n_ctx=4096 — larger context window",
        category="ssd_streaming", base="baseline",
        overrides={"n_ctx": 4096},
        hypothesis="Larger context increases KV cache pressure and may slow "
                   "expert streaming by reducing page cache headroom.",
        rationale="Quantifies KV cache vs expert cache memory trade-off.",
    ),
    Experiment(
        exp=10, title="n_ctx=1024 — moderate context",
        category="ssd_streaming", base="baseline",
        overrides={"n_ctx": 1024},
        hypothesis="1024-token context balances KV memory and generation length.",
        rationale="Intermediate data point between exp8 (512) and exp9 (4096).",
    ),

    # ── GPU Offloading / Metal (exp011–017) ────────────────────────────────
    Experiment(
        exp=11, title="n_gpu_layers=10 — partial Metal offload",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": 10},
        hypothesis="Offloading bottom 10 layers to Metal GPU reduces CPU compute "
                   "while SSD expert streaming remains on CPU.",
        rationale="Low n_gpu_layers has low overhead; tests whether Metal helps at all.",
    ),
    Experiment(
        exp=12, title="n_gpu_layers=20 — half layers on Metal",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": 20},
        hypothesis="Offloading half the layers to Metal should speed up attention "
                   "and non-expert compute.",
        rationale="Mid-point between baseline and full offload.",
    ),
    Experiment(
        exp=13, title="n_gpu_layers=-1 — all layers on Metal GPU",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": -1},
        hypothesis="Full Metal offload maximises GPU compute throughput; "
                   "expert weights still stream from SSD via mmap.",
        rationale="Tests peak Metal throughput for this MoE on 24 GB unified memory.",
    ),
    Experiment(
        exp=14, title="n_gpu_layers=-1 + flash_attn=True",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": -1, "flash_attn": True},
        hypothesis="Flash attention reduces attention memory bandwidth on GPU, "
                   "freeing more bandwidth for expert streaming.",
        rationale="Flash attn is most beneficial when combined with GPU offload.",
    ),
    Experiment(
        exp=15, title="n_gpu_layers=30 + flash_attn=True",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": 30, "flash_attn": True},
        hypothesis="Partial offload + flash attn may outperform full offload "
                   "if memory pressure limits full-GPU performance.",
        rationale="Tests whether partial offload + flash attn beats full offload.",
    ),
    Experiment(
        exp=16, title="n_gpu_layers=-1 + mlock=True",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": -1, "use_mlock": True},
        hypothesis="Full GPU offload with locked expert pages reduces SSD reads "
                   "during the decode loop.",
        rationale="Combines best GPU config with expert cache pinning.",
    ),
    Experiment(
        exp=17, title="n_gpu_layers=-1 + no mmap (all RAM)",
        category="gpu_offload", base="baseline",
        overrides={"n_gpu_layers": -1, "use_mmap": False},
        hypothesis="All weights in RAM + all layers on GPU: maximum throughput "
                   "if 24 GB can hold the model.",
        rationale="Upper bound on Metal performance, ignoring SSD streaming.",
    ),

    # ── CPU Threading (exp018–022) ──────────────────────────────────────────
    Experiment(
        exp=18, title="n_threads=1 — single thread",
        category="threading", base="baseline",
        overrides={"n_threads": 1},
        hypothesis="Single-thread baseline to measure threading overhead.",
        rationale="Lower bound; confirms multi-threading helps.",
    ),
    Experiment(
        exp=19, title="n_threads=4",
        category="threading", base="baseline",
        overrides={"n_threads": 4},
        hypothesis="4 threads should improve CPU-side dequant and attention.",
        rationale="Typical mid-range threading.",
    ),
    Experiment(
        exp=20, title="n_threads=8",
        category="threading", base="baseline",
        overrides={"n_threads": 8},
        hypothesis="8 threads fully utilises performance cores on M4.",
        rationale="M4 base has 4P cores; 8 may still help via HW threads.",
    ),
    Experiment(
        exp=21, title="n_threads=12",
        category="threading", base="baseline",
        overrides={"n_threads": 12},
        hypothesis="12 threads may exceed core count and add scheduling overhead.",
        rationale="Tests diminishing returns above CPU core count.",
    ),
    Experiment(
        exp=22, title=f"n_threads={_N_CPU} (all logical CPUs)",
        category="threading", base="baseline",
        overrides={"n_threads": _N_CPU},
        hypothesis="Using all logical CPUs saturates the CPU scheduler; "
                   "may compete with the Metal command queue.",
        rationale="Upper bound on threading; useful to find the sweet spot.",
    ),

    # ── Flash Attention (exp023–024) ────────────────────────────────────────
    Experiment(
        exp=23, title="flash_attn=True on CPU",
        category="attention", base="baseline",
        overrides={"flash_attn": True},
        hypothesis="Flash attention reduces attention memory traffic even on CPU, "
                   "potentially speeding up attention layers.",
        rationale="Tests flash_attn independently of GPU offload.",
    ),
    Experiment(
        exp=24, title="flash_attn=True + best thread count",
        category="attention", base="best",
        overrides={"flash_attn": True},
        hypothesis="Adding flash attn to the current best CPU config further "
                   "reduces attention overhead.",
        rationale="Stacking flash_attn on top of best threading config.",
    ),

    # ── Combo: Best GPU + Best Threading (exp025–029) ──────────────────────
    Experiment(
        exp=25, title="Best GPU layers + best n_threads",
        category="combo", base="best",
        overrides={},   # runner fills from current best + best_threads result
        hypothesis="The best GPU offload and best thread count stack additively.",
        rationale="Tests compounding of the two largest individual wins.",
    ),
    Experiment(
        exp=26, title="Best GPU + flash_attn + best threads",
        category="combo", base="best",
        overrides={"flash_attn": True},
        hypothesis="Adding flash_attn to the best GPU+threads combo.",
        rationale="Three-way stacking experiment.",
    ),
    Experiment(
        exp=27, title="Best GPU + mlock + flash_attn",
        category="combo", base="best",
        overrides={"use_mlock": True, "flash_attn": True},
        hypothesis="Locking pages + flash attn on top of best GPU config.",
        rationale="Tests whether page locking adds to GPU+flash_attn combo.",
    ),
    Experiment(
        exp=28, title="Best GPU + n_ctx=512 (small KV cache)",
        category="combo", base="best",
        overrides={"n_ctx": 512},
        hypothesis="Reducing KV cache with best GPU config frees memory for "
                   "expert page cache, improving hit rate.",
        rationale="KV cache memory pressure trade-off on the best so-far config.",
    ),
    Experiment(
        exp=29, title="Best GPU + n_batch=1024",
        category="combo", base="best",
        overrides={"n_batch": 1024},
        hypothesis="Larger batch on best GPU config improves prompt processing.",
        rationale="Batch size effect in context of GPU offload.",
    ),

    # ── Decoding strategies (exp030–034) ────────────────────────────────────
    Experiment(
        exp=30, title="Greedy decode (temp=0, top_k=1) — confirm baseline",
        category="decoding", base="best",
        overrides={"temperature": 0.0, "top_k": 1},
        hypothesis="Greedy decode should match or exceed sampled speed "
                   "as it skips probability computations.",
        rationale="Validates that greedy is the right choice for benchmarking.",
    ),
    Experiment(
        exp=31, title="Sampling (temp=0.7, top_k=50, top_p=0.9)",
        category="decoding", base="best",
        overrides={"temperature": 0.7, "top_k": 50, "top_p": 0.9},
        hypothesis="Sampling adds softmax + random sampling overhead; "
                   "expected to be slightly slower than greedy.",
        rationale="Quantifies sampling overhead on the best config.",
    ),
    Experiment(
        exp=32, title="top_k=1 (forced greedy via sampling)",
        category="decoding", base="best",
        overrides={"temperature": 0.3, "top_k": 1},
        hypothesis="Low temperature with top_k=1 approximates greedy with "
                   "negligible sampling cost.",
        rationale="Alternative greedy-like decoding variant.",
        expect_quality_risk=True,
    ),
    Experiment(
        exp=33, title="Longer generation (max_new_tokens=256)",
        category="decoding", base="best",
        overrides={"max_new_tokens": 256},
        hypothesis="Longer generation shows steady-state tok/s once KV cache "
                   "is warm and expert pages are cached.",
        rationale="Tests whether throughput improves as expert cache warms over a longer run.",
    ),
    Experiment(
        exp=34, title="Short generation (max_new_tokens=64)",
        category="decoding", base="best",
        overrides={"max_new_tokens": 64},
        hypothesis="Very short generations are TTFT-dominated; "
                   "measures TTFT vs steady-state throughput balance.",
        rationale="Opposite of exp33: TTFT-bound regime.",
    ),

    # ── Context length trade-offs (exp035–037) ──────────────────────────────
    Experiment(
        exp=35, title="n_ctx=768 — between 512 and 1024",
        category="context", base="best",
        overrides={"n_ctx": 768},
        hypothesis="768-token context may be the sweet spot between "
                   "KV memory pressure and generation length.",
        rationale="Fine-grained scan of the context/memory trade-off.",
    ),
    Experiment(
        exp=36, title="n_ctx=2048 + n_batch=256 combo",
        category="context", base="best",
        overrides={"n_ctx": 2048, "n_batch": 256},
        hypothesis="Default context with a moderate batch size.",
        rationale="Tests batch/context interaction on the best config.",
    ),
    Experiment(
        exp=37, title="n_ctx=512 + n_batch=256",
        category="context", base="best",
        overrides={"n_ctx": 512, "n_batch": 256},
        hypothesis="Minimal KV footprint + moderate batch: best memory efficiency.",
        rationale="Combination of smallest context + moderate batch.",
    ),

    # ── Reproducibility / stability (exp038–039) ────────────────────────────
    Experiment(
        exp=38, title="Seed stability check (3 identical runs)",
        category="reproducibility", base="best",
        overrides={},
        hypothesis="Three runs with the same seed and config should produce "
                   "the same tok/s within ±5% (noise floor characterisation).",
        rationale="Establishes measurement variance for all reported results.",
    ),
    Experiment(
        exp=39, title="Baseline re-run (cold recheck)",
        category="reproducibility", base="baseline",
        overrides={},
        hypothesis="Re-running the baseline after all experiments confirms "
                   "the hardware state hasn't changed (thermal, etc.).",
        rationale="Sanity check: baseline should match exp000 within ±5%.",
    ),

    # ── Final stacked best config (exp040) ──────────────────────────────────
    Experiment(
        exp=40, title="Final optimized stack (all validated wins combined)",
        category="combo", base="baseline",
        overrides={},   # runner fills from results/best_config.json
        hypothesis="The combined set of validated optimizations should deliver "
                   "the highest sustained tok/s observed in the experiment cycle.",
        rationale="exp040 is the project's headline result.",
    ),
]

# Fast lookup
BY_EXP: dict[int, Experiment] = {e.exp: e for e in EXPERIMENTS}


def get(exp: int) -> Experiment:
    if exp not in BY_EXP:
        raise KeyError(f"No experiment with exp={exp}")
    return BY_EXP[exp]
