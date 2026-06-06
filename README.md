# Flash-MoE / Inference-Engine Research

This repository contains two things kept deliberately separate:

## `inference-research/` — MSDS Project (main)

A complete autoresearch framework for local transformer inference optimization,
built for **Data Engineering 2 · Prof. Yue Chang**.

**Authors:** Mauricio Torres · Garret Knapp · Sammy Aridi

**Results (Apple M4 · 24 GB · MPS):**
- Baseline: 36.527 tok/s
- Best config (dynamic batching): 56.612 tok/s — **1.55× speedup**
- 42 experiments run, 8 kept, 34 discarded

```bash
cd inference-research
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

See [`inference-research/README.md`](inference-research/README.md) for full documentation.

---

## `reference/` — Original Flash-MoE Materials

The original Flash-MoE project that inspired this work: a pure C/Metal inference
engine running Qwen3.5-397B-A17B on a MacBook Pro at 4.4+ tok/s.

| Folder / File | Contents |
|---|---|
| `metal_infer/` | C + Metal compute engine source |
| `paper/` | Research paper (flash_moe.pdf) |
| `docs/` | Original experiment notes and pipeline plans |
| `MSDS_MoE/` | Course project spec (source of truth for inference-research/) |
| `results.tsv` | Original 58-experiment log |
| `progress.py` / `progress.png` | Results visualization |
| `repack_experts.py` | 4-bit expert weight packing |
| `expert_index.json` | Expert tensor manifest |
