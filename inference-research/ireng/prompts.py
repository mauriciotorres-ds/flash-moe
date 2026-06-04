"""Benchmark prompt suite.

Five categories required by the spec. Each prompt is fixed (no randomness) so
that runs are reproducible. `max_new_tokens` per category keeps total wall-time
predictable while still exercising the generation loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    system: str
    user: str
    max_new_tokens: int


_DEFAULT_SYS = "You are a concise, helpful assistant."

SUITE: List[Prompt] = [
    # ---- Factual / QA --------------------------------------------------
    Prompt("factual_1", "factual", _DEFAULT_SYS,
           "What is the capital of Australia, and why is it not Sydney?", 96),
    Prompt("factual_2", "factual", _DEFAULT_SYS,
           "List the planets of the solar system in order from the Sun.", 96),

    # ---- Coding --------------------------------------------------------
    Prompt("coding_1", "coding", "You are an expert Python programmer.",
           "Write a Python function `is_prime(n)` that returns True if n is prime. "
           "Include a short docstring.", 160),
    Prompt("coding_2", "coding", "You are an expert Python programmer.",
           "This code has a bug:\n\ndef avg(xs):\n    return sum(xs) / len(xs)\n\n"
           "Make it safe when xs is empty, and explain the fix in one sentence.", 160),

    # ---- Long reasoning ------------------------------------------------
    Prompt("reasoning_1", "reasoning", "You are a careful step-by-step reasoner.",
           "A train leaves city A at 9:00 traveling 60 km/h. Another leaves city B "
           "(180 km away) at 9:30 toward A at 90 km/h. At what time do they meet? "
           "Show your reasoning step by step.", 256),
    Prompt("reasoning_2", "reasoning", "You are a careful step-by-step reasoner.",
           "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops "
           "definitely Lazzies? Explain the logic carefully.", 200),

    # ---- Summarization -------------------------------------------------
    Prompt("summ_1", "summarization", "You summarize text faithfully and briefly.",
           "Summarize in 2 sentences:\n\nMixture-of-Experts (MoE) models activate "
           "only a subset of parameters per token, letting total parameter count "
           "grow without a proportional increase in per-token compute. A router "
           "selects K experts per layer; only those experts run. This makes very "
           "large models feasible on constrained hardware when expert weights are "
           "streamed from storage on demand.", 128),

    # ---- Structured output --------------------------------------------
    Prompt("struct_json", "structured", "You output strictly valid JSON, nothing else.",
           'Return a JSON object describing a book with keys: title, author, '
           'year (number), genres (array of strings). Use any real example.', 128),
    Prompt("struct_yaml", "structured", "You output valid YAML, nothing else.",
           "Produce a YAML config for a web server with: host, port, "
           "tls (enabled true/false), and a list of two allowed_origins.", 128),
    Prompt("struct_table", "structured", "You output a clean Markdown table.",
           "Make a 3-row Markdown table comparing CPU, GPU, and NPU on: "
           "parallelism, typical use, and power efficiency.", 160),
]

CATEGORIES = sorted({p.category for p in SUITE})


def by_category(cat: str) -> List[Prompt]:
    return [p for p in SUITE if p.category == cat]


def get(prompt_id: str) -> Prompt:
    for p in SUITE:
        if p.id == prompt_id:
            return p
    raise KeyError(prompt_id)
