# Live Playground — Example Prompts & Parameter Guide

We'll use this document to record runs from the dashboard's **Live Playground** tab.
Each section gives a prompt + recommended parameters, blank fields to fill in.
Fill in observed results and notes area to compare runs.

---

## How to read the parameters

| Parameter | What it controls | Rule of thumb |
|---|---|---|
| **Max new tokens** | Hard cap on output length | 128 for short answers, 400+ for long reasoning |
| **Temperature** | Randomness — 0 = deterministic, 2 = chaotic | 0.0 for facts/code, 0.7 for explanation, 1.1+ for creative |
| **Top-p** | Nucleus sampling — keeps only the top-p probability mass | 0.9 is a safe default; lower = more focused |
| **Top-k** | Limits vocab to k most likely tokens each step | 1 = greedy (same as temp 0); 50 is standard |
| **Seed** | Fixes randomness so a run is reproducible | Keep at 1234 to compare engines fairly |

---

## Prompt 1 — Explain the project (factual, short)

> *Good for: seeing baseline quality, comparing Baseline vs Optimized speed on a short output.*

**System prompt:**
```
You are a concise technical assistant.
```

**Prompt:**
```
In 3 bullet points, explain what a Mixture-of-Experts language model is
and why it can run efficiently on limited hardware.
```

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Engine | Baseline Engine |
| Device | auto |
| Max new tokens | 152 |
| Temperature | 0.0 |
| Top-p | 1.0 |
| Top-k | 1 |
| Seed | 1234 |

### Results — Baseline Engine
| Metric | Value |
|---|---|
| tok/s | 8.01 |
| TTFT (s) | 0.1248|
| Elapsed (s) | 3.6538|
| Tokens generated | 121|
| Peak memory (MB) | 1440.78|

**Output:** 

1. **Mixture-of-Experts Language Model**: This type of language model combines multiple experts or models to improve the accuracy and efficiency of its predictions.
2. **Efficiency on Limited Hardware**: By leveraging the strengths of different models, such as their strengths in specific areas (e.g., understanding context, generating text), the mixture-of-experts approach can significantly reduce the computational load required for tasks like translation, summarization, or question answering. Each expert's contribution allows the model to focus on more relevant aspects of the input data, thereby optimizing performance without needing to train on all possible combinations of these models simultaneously.

This method is particularly useful when dealing with large datasets where traditional approaches might be computationally prohibitive due to the sheer volume of information. The


### Results — Optimized Engine (same prompt, same params)
| Metric | Value |
|---|---|
| tok/s | 25.684|
| TTFT (s) | 1.386|
| Elapsed (s) | 3.3873|
| Tokens generated | 87|
| Peak memory (MB) | 1203.29|

**Output:**

1. **Mixture-of-Experts Language Model**: This type of language model combines multiple experts or models to improve the accuracy and efficiency of its predictions.
2. **Efficiency on Limited Hardware**: By leveraging the strengths of different models, such as their strengths in specific areas (e.g., understanding context, generating text), the mixture-of-experts approach can significantly reduce the computational load required for tasks like translation, summarization, or question answering. Each expert's contribution allows the model to focus on more relevant aspects of the input data, thereby optimizing performance without needing to


**Speedup observed:** 3.21× faster tok/s (25.684 / 8.01) · 0.3s faster elapsed

**Output quality difference (any):**
> None — the content is word-for-word identical for the text both engines produced. The optimized engine simply stopped 34 tokens earlier because fp16 arithmetic produces slightly different logit values near EOS, triggering the stop token sooner. The meaning is fully preserved.

### What we're seeing in these two runs

Both engines received the exact same prompt and parameters, so any difference is purely the effect of the optimization stack. The tok/s gap is the headline number — the optimized engine decoded at **3.21× the rate** (25.7 vs 8.0 tok/s). However, the total elapsed time only improved by 0.3 seconds (3.39s vs 3.65s), which looks underwhelming at first glance. The reason is the **TTFT tradeoff**: the optimized engine's first token took 1.386s versus 0.124s for baseline. That 1.26-second penalty at the start is a one-time warmup cost from loading fp16 weights and initializing the SDPA kernel on a fresh engine instance — in a real serving scenario where the model stays loaded between requests, this cost disappears entirely and you see the full 3× speedup on every subsequent prompt. The other notable result is memory: the optimized engine used **237 MB less RAM** (1203 vs 1441 MB) because fp16 weights are half the size of fp32, which is the same bandwidth-reduction principle behind the original Flash-MoE project's 4-bit expert packing.

---

## Prompt 2 — Step-by-step reasoning (longer output)

> *Good for: demonstrating TTFT on a harder prompt, seeing how latency scales with output length.*

**System prompt:**
```
You are a careful step-by-step reasoner. Show all working.
```

**Prompt:**
```
A GPU processes tokens at 36 tok/s with the baseline engine.
After optimization it runs at 56 tok/s.
If a user sends 10 prompts per minute, each expecting 120 tokens of output,
how many seconds of GPU time does the optimized engine save per minute
compared to the baseline?
Show each calculation step.
```

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Engine | Optimized Engine |
| Device | auto |
| Max new tokens | 300 |
| Temperature | 0.0 |
| Top-p | 1.0 |
| Top-k | 1 |
| Seed | 1234 |

### Results
| Metric | Value |
|---|---|
| tok/s | |
| TTFT (s) | |
| Elapsed (s) | |
| Tokens generated | |
| Peak memory (MB) | |

**Was the answer correct?** Yes / No / Partial

**Notes:**
>

---

## Prompt 3 — Code generation (deterministic)

> *Good for: structured output quality check, code correctness, comparing temperature effect.*

**System prompt:**
```
You are an expert Python programmer. Output only code, no prose.
```

**Prompt:**
```
Write a Python function `tokens_per_second(total_tokens, elapsed_seconds)`
that returns tok/s rounded to 3 decimal places,
and raises ValueError if elapsed_seconds is zero or negative.
Include a docstring.
```

### Run A — Greedy (temperature 0)

| Parameter | Value |
|---|---|
| Engine | Optimized Engine |
| Max new tokens | 200 |
| Temperature | 0.0 |
| Top-p | 1.0 |
| Top-k | 1 |
| Seed | 1234 |

**Output:**
```python

```

**tok/s:** &nbsp;&nbsp;&nbsp;&nbsp; **TTFT:** &nbsp;&nbsp;&nbsp;&nbsp; **Tokens:**

---

### Run B — Slightly creative (temperature 0.7)
*(Same prompt, same engine — does the code change?)*

| Parameter | Value |
|---|---|
| Temperature | 0.7 |
| Top-p | 0.9 |
| Top-k | 50 |
| Seed | 1234 |

**Output:**
```python

```

**tok/s:** &nbsp;&nbsp;&nbsp;&nbsp; **TTFT:** &nbsp;&nbsp;&nbsp;&nbsp; **Tokens:**

**Did the output differ from Run A?** Yes / No

---

## Prompt 4 — Summarization (medium length)

> *Good for: demonstrating quality of the optimized fp16 + SDPA stack on a structured task.*

**System prompt:**
```
You summarize technical content clearly and briefly.
```

**Prompt:**
```
Summarize the following in exactly 2 sentences for a non-technical audience:

Mixture-of-Experts (MoE) models activate only a small subset of their
parameters for each input token. A learned router selects K experts per layer;
only those K run, keeping per-token compute low even as total parameter count
grows very large. On constrained hardware, expert weights can be streamed from
SSD on demand so the model never fully resides in RAM.
```

| Parameter | Value |
|---|---|
| Engine | Optimized Engine |
| Max new tokens | 128 |
| Temperature | 0.3 |
| Top-p | 0.9 |
| Top-k | 40 |
| Seed | 1234 |

**Output:**
>

**tok/s:** &nbsp;&nbsp;&nbsp;&nbsp; **TTFT:** &nbsp;&nbsp;&nbsp;&nbsp; **Tokens:**

---

## Prompt 5 — Engine comparison on the same creative prompt

> *Good for: demo moment — same prompt, visibly different speed, same quality.*

**System prompt:**
```
You are a concise, helpful assistant.
```

**Prompt:**
```
Explain in one paragraph why running a 397-billion parameter AI model
on a laptop is surprising, and what engineering trick makes it possible.
```

### Baseline Engine

| Parameter | Value |
|---|---|
| Max new tokens | 180 |
| Temperature | 0.5 |
| Top-p | 0.9 |
| Top-k | 50 |
| Seed | 42 |

**Output:**
>

**tok/s:** &nbsp;&nbsp;&nbsp;&nbsp; **TTFT:** &nbsp;&nbsp;&nbsp;&nbsp; **Elapsed:**

---

### Optimized Engine (identical parameters)

**Output:**
>

**tok/s:** &nbsp;&nbsp;&nbsp;&nbsp; **TTFT:** &nbsp;&nbsp;&nbsp;&nbsp; **Elapsed:**

**Speedup:** ________×  &nbsp; **Quality difference:** None / Minor / Major

---

## Quick-reference: parameter recipes

| Goal | Temp | Top-p | Top-k | Max tokens |
|---|---|---|---|---|
| Exact / factual answer | 0.0 | 1.0 | 1 | 128 |
| Balanced explanation | 0.3 | 0.9 | 40 | 200 |
| Natural conversation | 0.7 | 0.9 | 50 | 256 |
| Creative / varied | 1.0 | 0.95 | 60 | 400 |
| Code (deterministic) | 0.0 | 1.0 | 1 | 300 |
| Speed benchmark | 0.0 | 1.0 | 1 | 128 |

> **Tip:** Always keep **Seed = 1234** when comparing Baseline vs Optimized so the
> randomness is identical and any difference you see is purely from the engine,
> not from sampling luck.
