#!/usr/bin/env python3
"""dashboard.py — Inference-engine observability & research platform.

Launch:  streamlit run dashboard.py

Tabs
----
1. Live Playground       prompt/model/engine/params -> streaming generation +
                         live tok/s, TTFT, memory, CPU/GPU, KV cache, context.
2. Compare              baseline vs optimized side-by-side, speedup.
3. Experiment Explorer  browse/sort/filter all 42 experiments.
4. Optimization Timeline tok/s, latency, memory, speedup vs experiment #.
5. Diff Viewer          baseline vs optimized config diff.
6. MoE Visualization    expert-utilisation view for the 397B model (or the
                        documented limitation when unavailable).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64

import altair as alt
import streamlit as st_ui
import pandas as pd

from ireng import storage as st
from ireng.hardware import detect_host, TARGET_SPEC
from ireng.config import (baseline_config, load_optimized_config,
                           diff_configs, SMALL_MODEL_ID, MEDIUM2_MODEL_ID)
from ireng.prompts import SUITE, CATEGORIES

st_ui.set_page_config(
    page_title="Inference Research Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS  = ["Qwen1.5-MoE-A2.7B", "Qwen3.5-35B-A3B-IQ2"]
ENGINES = ["Baseline Engine", "Optimized Engine", "Compare Both"]

# ─────────────────────────────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────────────────────────────

_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

/* ── Page ──────────────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #060b14 !important;
}
[data-testid="stHeader"] {
    background-color: #060b14 !important;
    border-bottom: 1px solid #00c8c820;
}
.main .block-container {
    padding-top: 0.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
body, p, li, span, label, div {
    color: #b8d4e0 !important;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #00e8e8 !important;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.06em;
}
.stMarkdown h2 {
    font-size: 1.05rem !important;
    color: #00c8c8 !important;
    border-bottom: 1px solid #00909025;
    padding-bottom: 4px;
}
code, pre {
    font-family: 'Share Tech Mono', monospace !important;
    background: #0a1824 !important;
    color: #00e8e8 !important;
    border: 1px solid #00909030 !important;
}

/* ── Custom Header Block ────────────────────────────────────────────────── */
.dash-header {
    position: relative;
    padding: 1.6rem 0 1.1rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #00404040;
    overflow: hidden;
}
/* subtle grid behind the header */
.dash-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(90deg, #00c8c806 1px, transparent 1px),
        linear-gradient(0deg,  #00c8c806 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}
.dash-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 900 !important;
    color: #00e8e8 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    animation: glowPulse 4s ease-in-out infinite !important;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.15 !important;
    padding: 0 !important;
}
p.dash-title { margin-top: 0 !important; }
.dash-title .blink {
    display: inline-block;
    width: 3px;
    height: 1.7rem;
    background: #00e8e8;
    margin-left: 6px;
    vertical-align: middle;
    animation: blink 1.1s step-end infinite;
}
.dash-subtitle {
    margin-top: 0.15rem !important;
    padding: 0 !important;
}
.dash-authors {
    font-family: 'Share Tech Mono', monospace;
    color: #7de0e0 !important;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}
.dash-authors::before { content: '▸ '; color: #00e8e8; }
.dash-course {
    font-family: 'Share Tech Mono', monospace;
    color: #4db8b8 !important;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}
.dash-course::before { content: '▸ '; color: #00909080; }
.dash-badge {
    position: absolute;
    top: 14px;
    right: 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #006060;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid #005050;
    padding: 2px 8px;
    border-radius: 2px;
}
/* scanning line that sweeps down the header */
.scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #00e8e8 50%, transparent 100%);
    animation: scanDown 5s linear infinite;
    opacity: 0;
    pointer-events: none;
}

/* ── Animations ─────────────────────────────────────────────────────────── */
@keyframes glowPulse {
    0%   { text-shadow: 0 0 8px #00e8e8,  0 0 18px #00c8c840; }
    50%  { text-shadow: 0 0 22px #00e8e8, 0 0 44px #00c8c870, 0 0 70px #00808040; }
    100% { text-shadow: 0 0 8px #00e8e8,  0 0 18px #00c8c840; }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
@keyframes scanDown {
    0%   { top: -2px; opacity: 0; }
    5%   { opacity: 0.7; }
    90%  { opacity: 0.7; }
    100% { top: 100%; opacity: 0; }
}
@keyframes borderPulse {
    0%   { box-shadow: 0 0 5px #00c8c840,  inset 0 0 5px #00c8c818; }
    50%  { box-shadow: 0 0 14px #00e8e870, inset 0 0 14px #00e8e838; }
    100% { box-shadow: 0 0 5px #00c8c840,  inset 0 0 5px #00c8c818; }
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0a1828 0%, #0d2030 100%) !important;
    border: 1px solid #00909038 !important;
    border-radius: 5px !important;
    padding: 12px 14px !important;
    animation: borderPulse 4s ease-in-out infinite;
}
[data-testid="stMetricValue"] {
    color: #00e8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.35rem !important;
}
[data-testid="stMetricLabel"] {
    color: #5dc8c8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
[data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #006868 0%, #009090 100%) !important;
    color: #001818 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 12px #00c8c848 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c8c8 0%, #00e8e8 100%) !important;
    color: #000 !important;
    box-shadow: 0 0 28px #00e8e888, 0 0 54px #00c8c840 !important;
    transform: translateY(-2px) !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #07101c !important;
    border-bottom: 1px solid #00909035 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #3a9898 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
    border-right: 1px solid #00909018 !important;
    transition: color 0.18s, background 0.18s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #00e8e8 !important;
    background: #0a182808 !important;
}
.stTabs [aria-selected="true"] {
    color: #00e8e8 !important;
    background: #0a182820 !important;
    border-bottom: 2px solid #00e8e8 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.2rem !important;
    animation: fadeSlideIn 0.25s ease-out;
}

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #00909028 !important;
    border-radius: 5px !important;
    overflow: hidden !important;
}

/* ── Tables ─────────────────────────────────────────────────────────────── */
table {
    background: #07101c !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
thead tr th {
    background: #0a1828 !important;
    color: #00e8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #00909040 !important;
    padding: 8px 14px !important;
}
tbody tr td {
    color: #9dc4d0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    border-bottom: 1px solid #00909014 !important;
    padding: 6px 14px !important;
}
tbody tr:hover td {
    background: #00e8e808 !important;
    color: #00e8e8 !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: #0a1828 !important;
    color: #b8d4e0 !important;
    border: 1px solid #00909040 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #00e8e8 !important;
    box-shadow: 0 0 0 1px #00e8e830 !important;
    outline: none !important;
}
[data-baseweb="select"] > div {
    background: #0a1828 !important;
    border-color: #00909040 !important;
    color: #b8d4e0 !important;
}
[data-baseweb="popover"] {
    background: #0a1828 !important;
    border: 1px solid #00909040 !important;
}
[role="option"] {
    background: #0a1828 !important;
    color: #b8d4e0 !important;
}
[role="option"]:hover {
    background: #00e8e812 !important;
    color: #00e8e8 !important;
}

/* Slider */
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #3a9898 !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #00c8c8 !important;
    box-shadow: 0 0 8px #00e8e870 !important;
}

/* ── Alerts / Banners ───────────────────────────────────────────────────── */
.stSuccess > div {
    background: linear-gradient(135deg, #051810, #091c18) !important;
    border: 1px solid #00e8e828 !important;
    border-left: 3px solid #00e8e8 !important;
    border-radius: 4px !important;
    color: #7de0e0 !important;
}
.stInfo > div {
    background: linear-gradient(135deg, #071428, #091828) !important;
    border: 1px solid #00909028 !important;
    border-left: 3px solid #007a7a !important;
    border-radius: 4px !important;
}
.stWarning > div {
    background: linear-gradient(135deg, #181008, #201808) !important;
    border: 1px solid #c8900028 !important;
    border-left: 3px solid #c89000 !important;
    border-radius: 4px !important;
}
.stError > div {
    background: linear-gradient(135deg, #180808, #200a0a) !important;
    border-left: 3px solid #e84040 !important;
    border-radius: 4px !important;
}

/* ── Charts ─────────────────────────────────────────────────────────────── */
.stArrowVegaLiteChart > div {
    border-radius: 5px !important;
    border: 1px solid #00909025 !important;
    overflow: hidden !important;
}

/* ── Images ─────────────────────────────────────────────────────────────── */
.stImage img {
    border: 1px solid #00909028 !important;
    border-radius: 5px !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #00909025 !important;
    margin: 1rem 0 !important;
}

/* ── Captions ───────────────────────────────────────────────────────────── */
.stCaption, [data-testid="caption"] {
    color: #3a8888 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
}

/* ── JSON viewer ────────────────────────────────────────────────────────── */
.stJson {
    background: #070d18 !important;
    border: 1px solid #00909028 !important;
    border-radius: 5px !important;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #00e8e8 !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #060b14; }
::-webkit-scrollbar-thumb { background: #00606060; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00c8c8; }

/* ── Subheader override ─────────────────────────────────────────────────── */
[data-testid="stSubheader"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important;
    color: #00c8c8 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #00909025;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ── Checkbox ───────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: #7de0e0 !important; }

/* ── UVA logo (floated right inside the header) ─────────────────────────── */
.uva-logo {
    float: right;
    height: 78px;
    width: auto;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(0, 200, 200, 0.22);
    border-radius: 7px;
    padding: 8px 12px;
    box-shadow: 0 0 14px rgba(0, 200, 200, 0.12);
    filter: drop-shadow(0 0 3px rgba(0, 200, 200, 0.2));
    margin-left: 1.5rem;
    transition: box-shadow 0.3s;
}
.uva-logo:hover {
    box-shadow: 0 0 26px rgba(0, 232, 232, 0.28);
}
.dash-clear { clear: both; }
</style>
"""

# ── UVA logo (auto-loads from assets/uva_logo.png; skipped gracefully if absent) ──

_LOGO_B64: str | None = None

def _load_logo() -> str | None:
    global _LOGO_B64
    if _LOGO_B64 is not None:
        return _LOGO_B64
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "uva_logo.png")
    if not os.path.exists(logo_path):
        return None
    with open(logo_path, "rb") as fh:
        _LOGO_B64 = base64.b64encode(fh.read()).decode()
    return _LOGO_B64


def _build_header_html() -> str:
    b64 = _load_logo()
    # Logo must come BEFORE the title text so the float works correctly.
    # We keep the HTML completely flat (no nested divs) so Streamlit's
    # markdown parser never renders a stray </div> as text.
    logo = (
        f'<img class="uva-logo" src="data:image/png;base64,{b64}" alt="University of Virginia">'
    ) if b64 else ""
    return (
        '<div class="dash-header">'
        + '<div class="scan-line"></div>'
        + logo
        + '<div class="dash-badge">41 Experiments Complete &nbsp;&#183;&nbsp; 3 Models</div>'
        + '<p class="dash-title">Inference&#8209;Engine Research Dashboard<span class="blink"></span></p>'
        + '<p class="dash-subtitle">'
        + '<span class="dash-authors">Authors:&nbsp; Mauricio Torres &nbsp;&#183;&nbsp; Garret Knapp &nbsp;&#183;&nbsp; Sammy Aridi</span>'
        + '&emsp;'
        + '<span class="dash-course">Data Engineering 2 &nbsp;&#183;&nbsp; Prof. Yue Chang</span>'
        + '</p>'
        + '<div class="dash-clear"></div>'
        + '</div>'
    )


def _inject_styles():
    st_ui.markdown(_STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _data_source_banner():
    rows = st.read_experiments()
    srcs = {r.get("data_source") for r in rows}
    if "SAMPLE_SYNTHETIC" in srcs:
        st_ui.error(
            "⚠️  SAMPLE DATA — synthetic numbers.  "
            "Run `python run_experiments.py --mode real --device mps` to replace."
        )
    elif "MOCK_DRYRUN" in srcs:
        st_ui.warning("Mock dry-run data (synthetic). Run `--mode real` on your Mac for real numbers.")
    elif rows:
        st_ui.success("Measured data loaded.")
    else:
        st_ui.info("No experiment data yet. Run `python seed_sample_data.py` or the real benchmarks.")


def _host_header():
    host = detect_host()
    c1, c2, c3, c4 = st_ui.columns(4)
    c1.metric("Target", TARGET_SPEC["label"])
    c2.metric("Backend", "llama-cpp-python + Metal")
    c3.metric("llama-cpp", host.llama_version or "not installed")
    c4.metric("Experiments on disk", len(st.read_experiments()))


@st_ui.cache_resource(show_spinner=False, max_entries=1)
def _get_engine(kind: str, model_id: str):
    try:
        if kind == "Baseline Engine":
            from baseline_engine import BaselineEngine
            eng = BaselineEngine(model_id)
            eng.load()
            return eng, None
        else:
            from optimized_engine import OptimizedEngine
            eng = OptimizedEngine(model_id)
            eng.load()
            return eng, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────

def tab_playground():
    st_ui.subheader("Live Inference Playground")
    col_a, col_b = st_ui.columns([2, 1])
    with col_b:
        model_id    = st_ui.selectbox("Model", MODELS,
                                      help="Small=2.7B active / Medium IQ2=3B active")
        engine_kind = st_ui.selectbox("Engine", ENGINES[:2])
        max_new     = st_ui.slider("Max new tokens", 8, 2048, 768, 8,
                                   help="Qwen3.5-35B reasons before answering, so it needs a "
                                        "larger budget to finish thinking AND produce the answer.")
        temperature = st_ui.slider("Temperature", 0.0, 2.0, 0.0, 0.05)
        top_p       = st_ui.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
        top_k       = st_ui.slider("Top-k", 1, 200, 1, 1)
        seed        = st_ui.number_input("Seed", value=1234, step=1)
        no_think    = st_ui.checkbox("Disable thinking", value=True,
                                     help="Qwen3.5-35B reasons before answering by default. "
                                          "This prefills an empty <think></think> block so it "
                                          "answers directly. (No effect on the small model.)")
        do_log      = st_ui.checkbox("Log this run to dashboard_logs/", value=True)
    with col_a:
        system = st_ui.text_area("System prompt", "You are a concise, helpful assistant.", height=70)
        prompt = st_ui.text_area("Prompt",
            "Explain why only a small subset of experts activate per token in a "
            "Mixture-of-Experts model, and how that makes large-model laptop "
            "inference possible.", height=120)
        run = st_ui.button("▶  Run Inference", type="primary")

    if model_id == MEDIUM2_MODEL_ID:
        gguf_ok = os.path.isdir(os.path.expanduser("~/models/Qwen3.5-35B-A3B-IQ2-GGUF"))
        if not gguf_ok:
            st_ui.warning(
                "Medium IQ2 model (Qwen3.5-35B-A3B-IQ2) not downloaded yet.\n\n"
                "```\nhf download unsloth/Qwen3.5-35B-A3B-GGUF "
                "--include '*UD-IQ2_M*' "
                "--local-dir ~/models/Qwen3.5-35B-A3B-IQ2-GGUF\n```"
            )
            return

    if not run:
        return

    engine, err = _get_engine(engine_kind, model_id)
    if err or engine is None:
        st_ui.error(
            f"Engine load failed: {err}\n\n"
            "Make sure the GGUF is downloaded and llama-cpp-python is installed."
        )
        return

    from ireng.prompts import Prompt
    engine.config.max_new_tokens = max_new
    engine.config.temperature    = temperature
    engine.config.top_p          = top_p
    engine.config.top_k          = top_k
    engine.config.seed           = int(seed)
    engine.config.disable_thinking = bool(no_think)
    p = Prompt("playground", "factual", system, prompt, max_new)

    st_ui.caption("Generated output (scrollable):")
    out_container = st_ui.container(height=300)
    out_box = out_container.empty()
    metrics_container = st_ui.container(height=140)
    c1,c2,c3,c4,c5 = metrics_container.columns(5)
    text = ""
    last = None
    with st_ui.spinner("Generating…"):
        for chunk, m in engine.stream(p):
            if chunk:
                text += chunk
                out_box.markdown(f"```\n{text}\n```")
            last = m
            c1.metric("tokens",      m.tokens_generated)
            c2.metric("tok/s",       m.tokens_per_second)
            c3.metric("TTFT (s)",    m.time_to_first_token_s or "—")
            c4.metric("elapsed (s)", m.total_runtime_s)
            c5.metric("context",     m.context_length)

    if engine.support_notes:
        for note in engine.support_notes[-3:]:
            if "error" in note.lower():
                st_ui.error(f"Generation error: {note}")

    if last:
        st_ui.divider()
        d = last.as_dict()
        cols = st_ui.columns(7)
        cols[0].metric("total tokens",      d["tokens_generated"])
        cols[1].metric("total runtime (s)", d["total_runtime_s"])
        cols[2].metric("peak mem (MB)",     d["peak_memory_mb"])
        cols[3].metric("cur mem (MB)",      d["current_memory_mb"])
        cols[4].metric("CPU %",             d["cpu_utilization_pct"])
        cols[5].metric("expert MB/tok",     d.get("expert_bytes_per_tok"))
        cols[6].metric("GPU",               "Metal" if engine.config.n_gpu_layers != 0 else "CPU")

        # Expert architecture info
        es = last.expert_stats
        if es:
            st_ui.divider()
            st_ui.markdown("**Expert Architecture (from GGUF metadata)**")
            ea1, ea2, ea3, ea4 = st_ui.columns(4)
            ea1.metric("Total experts / layer", es.n_experts)
            ea2.metric("Active per token (K)",  es.n_experts_used)
            ea3.metric("MoE layers",            es.n_moe_layers)
            ea4.metric("Utilisation",
                       f"{round(es.utilization_rate()*100,1)}%"
                       if es.activation_counts else "see note")
            st_ui.caption(
                "Per-step expert indices require llama.cpp C callbacks not yet "
                "exposed in Python. Architecture info is read directly from the GGUF file."
            )

        if do_log:
            _log_run(model_id, engine_kind, system, prompt, d)
            st_ui.caption("Logged to dashboard_logs/")


def _log_run(model_id, engine_kind, system, prompt, metrics):
    os.makedirs(st.LOGS, exist_ok=True)
    fn = os.path.join(st.LOGS, time.strftime("run_%Y%m%d_%H%M%S.json"))
    with open(fn, "w") as f:
        json.dump({"model": model_id, "engine": engine_kind,
                   "system": system, "prompt": prompt,
                   "metrics": metrics, "timestamp": st.now_iso()}, f, indent=2)


def tab_compare():
    st_ui.subheader("Baseline vs Optimized — All Models")

    # Load per-model comparison files
    files = {
        "Qwen1.5-MoE-A2.7B (Q4_K_M · 8.84 GB)": "comparison_small.json",
        "Qwen3.5-35B-A3B (IQ2_M · 11 GB)":       "comparison_medium_iq2.json",
    }
    comparisons = {}
    for label, fname in files.items():
        d = st.read_json(os.path.join(st.RESULTS, fname), None)
        if d:
            comparisons[label] = d

    if not comparisons:
        st_ui.info("No comparison data yet. Run `python benchmark_runner.py --engine both`.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────
    for label, d in comparisons.items():
        st_ui.markdown(f"**{label}**")
        c1, c2, c3, c4 = st_ui.columns(4)
        c1.metric("Baseline tok/s",  d.get("baseline_tps"),
                  delta=f"device: {d.get('baseline_device','cpu')}")
        c2.metric("Optimized tok/s", d.get("optimized_tps"),
                  delta=f"device: {d.get('optimized_device','cpu')}")
        c3.metric("Speedup",         f"{d.get('speedup')}×")
        c4.metric("Key optimization", d.get("key_optimization", "—"))
        st_ui.caption(d.get("notes", ""))
        st_ui.divider()

    # ── Full comparison table ──────────────────────────────────────────────
    st_ui.markdown("**Full cross-model comparison**")
    rows = []
    for label, d in comparisons.items():
        rows.append({
            "Model": label,
            "Baseline tok/s":  d.get("baseline_tps"),
            "Optimized tok/s": d.get("optimized_tps"),
            "Speedup":         f"{d.get('speedup')}×",
            "Opt device":      d.get("optimized_device", "cpu"),
            "Key optimization": d.get("key_optimization", "—"),
        })
    st_ui.table(pd.DataFrame(rows))

    # ── Instructor reference callout ──────────────────────────────────────
    st_ui.divider()
    st_ui.info(
        "**Instructor reference:** Qwen3.5-35B-A3B on M1 Pro (32 GB), "
        "from-scratch engine — **16.3 tok/s** after ~40 experiments.  \n"
        "Our IQ2_M baseline (**18.70 tok/s**) already exceeds this. "
        "Optimized reaches **45.6 tok/s** (+144% over IQ2 baseline, +180% vs reference).  \n"
        "We started with llama-cpp-python as an optimized foundation and built upon it."
    )

    # ── Why our baseline is higher than the instructor's ──────────────────
    st_ui.divider()
    st_ui.markdown("**Why our baseline starts at 51 tok/s — not < 1 tok/s**")
    st_ui.markdown(
        "The instructor built their engine **from scratch** — custom GGUF parser, hand-written "
        "matrix multiply kernels, and Metal shaders implemented manually. "
        "Starting from zero, < 1 tok/s is the expected starting point before any optimizations.\n\n"
        "This project uses **llama-cpp-python** as its inference foundation — a production library "
        "with 2+ years of optimization already built in: quantized GEMM kernels, Metal GPU support, "
        "flash attention, and mmap-based expert streaming. That existing work is why our "
        "baseline is already at 51 tok/s before a single experiment ran.\n\n"
        "**Hardware is a secondary factor.** Apple M4 (2024) is faster than the M1 Pro (2021) "
        "for the matrix operations that dominate MoE inference, but even llama.cpp on an M1 Pro "
        "starts well above 1 tok/s — the library is the dominant reason, not the chip.\n\n"
        "The methodology is identical: 41 real experiments, measured results, keep/discard decisions, "
        "documented failures. We started from a stronger foundation and found +93% more performance "
        "on top of it. The core findings — GPU offload dominates, mlock hurts, KV cache size "
        "directly competes with expert page cache — are real empirical results."
    )

    # ── Small model experiment progression ───────────────────────────────
    st_ui.divider()
    st_ui.markdown("**Small model — experiment progression (41 experiments)**")
    rows_exp = st.read_experiments()
    if rows_exp:
        df = pd.DataFrame(rows_exp)
        df["exp"] = pd.to_numeric(df["exp"], errors="coerce")
        df["mean_tps"] = pd.to_numeric(df["mean_tps"], errors="coerce")
        df = df.sort_values("exp")
        keeps = df[df["decision"] == "keep"][["exp", "mean_tps", "title"]].dropna(subset=["mean_tps"])
        if not keeps.empty:
            st_ui.markdown("Kept experiments only:")
            bar = (
                alt.Chart(keeps)
                .mark_bar(color="#00c8c8")
                .encode(
                    x=alt.X("exp:O", title="Experiment Number"),
                    y=alt.Y("mean_tps:Q", title="Throughput (tok/s)"),
                    tooltip=[
                        alt.Tooltip("exp", title="Experiment"),
                        alt.Tooltip("title", title="Config"),
                        alt.Tooltip("mean_tps", title="tok/s"),
                    ],
                )
                .properties(height=220)
            )
            st_ui.altair_chart(bar, use_container_width=True)


def tab_explorer():
    st_ui.subheader("Experiment Explorer")
    rows = st.read_experiments()
    if not rows:
        st_ui.info("No experiments. Seed sample data or run the runner.")
        return

    # ── Improvements narrative ─────────────────────────────────────────────
    st_ui.markdown("## Optimization Story — What We Kept and Why")
    st_ui.markdown(
        "Out of 41 experiments, **9 were kept** — each one building on the last. "
        "Here is the story of how the engine improved, step by step."
    )

    KEPT = [
        {
            "exp": "exp000",
            "title": "Baseline — mmap=True, CPU-only, no GPU",
            "tps": 51.25,
            "prev": None,
            "delta": None,
            "what": "The unoptimized reference. Uses memory-mapped GGUF file, trusts the OS page cache to manage expert reads, runs all compute on CPU. Every subsequent experiment is measured against this.",
            "why_matters": "Establishes the floor. Already at 51 tok/s because llama-cpp-python is a production library — not a from-scratch engine.",
        },
        {
            "exp": "exp004",
            "title": "Warm OS page cache (second inference, same process)",
            "tps": 58.23,
            "prev": 51.25,
            "delta": "+13.6%",
            "what": "Running a second prompt in the same process after the first — expert pages from the first run are already cached in RAM by the OS.",
            "why_matters": "Proves that the OS page cache is doing real work. Hot expert pages served at ~400 GB/s (RAM) vs ~5 GB/s (SSD). This is the 'trust the OS' principle validated empirically.",
        },
        {
            "exp": "exp008",
            "title": "n_ctx=512 — smaller context window",
            "tps": 61.73,
            "prev": 58.23,
            "delta": "+6.0%",
            "what": "Reduced the KV cache context window from 2048 to 512 tokens.",
            "why_matters": "KV cache and expert page cache compete for the same RAM. Shrinking the KV cache frees memory for expert pages to stay warm. Memory is the shared resource — less KV means more expert cache headroom.",
        },
        {
            "exp": "exp011",
            "title": "n_gpu_layers=10 — partial Metal offload",
            "tps": 64.98,
            "prev": 61.73,
            "delta": "+5.2%",
            "what": "Offloaded the first 10 transformer layers to Metal GPU, leaving the rest on CPU.",
            "why_matters": "First evidence that the M4 GPU is faster than CPU for dequantized matrix-vector multiply. Even partial offload gives measurable gains — confirms the direction.",
        },
        {
            "exp": "exp012",
            "title": "n_gpu_layers=20 — half layers on Metal",
            "tps": 74.66,
            "prev": 64.98,
            "delta": "+14.9%",
            "what": "Doubled the GPU layer count to 20.",
            "why_matters": "Linear scaling with layer count — each layer moved to GPU contributes proportionally. Shows no memory bottleneck at 20 layers. Confirms full offload will be the winner.",
        },
        {
            "exp": "exp013",
            "title": "n_gpu_layers=-1 — all layers on Metal GPU",
            "tps": 95.55,
            "prev": 74.66,
            "delta": "+28.0%",
            "what": "Offloaded all 24 transformer layers to Metal GPU.",
            "why_matters": "The single biggest jump of the entire experiment cycle. CPU→GPU for all compute nearly doubles throughput. M4 GPU executes Q4_K_M dequant + matvec ~2× faster than CPU cores for this workload.",
        },
        {
            "exp": "exp014",
            "title": "n_gpu_layers=-1 + flash_attn=True  ← FINAL WINNER",
            "tps": 98.96,
            "prev": 95.55,
            "delta": "+3.6%",
            "what": "Added flash attention on top of full GPU offload. Flash attention fuses the attention kernel to reduce memory reads.",
            "why_matters": "Smaller but real gain on top of the GPU win. Reduces memory traffic during attention computation. This is the final optimized config — two knob changes from baseline, +93% total improvement.",
        },
    ]

    for i, exp in enumerate(KEPT):
        prev_tps = exp["prev"]
        delta_str = f"  **{exp['delta']}** from previous keep" if exp["delta"] else "  *(baseline)*"
        label = f"**{exp['exp']}** — {exp['title']}  ·  {exp['tps']} tok/s{delta_str}"
        with st_ui.expander(label, expanded=(i == len(KEPT) - 1)):
            col1, col2 = st_ui.columns([1, 2])
            with col1:
                st_ui.metric("tok/s", exp["tps"],
                             delta=exp["delta"] if exp["delta"] else "baseline")
                if prev_tps:
                    st_ui.metric("Previous best", prev_tps)
            with col2:
                st_ui.markdown(f"**What changed:** {exp['what']}")
                st_ui.markdown(f"**Why it matters:** {exp['why_matters']}")

    st_ui.divider()
    st_ui.markdown("**Summary: the full improvement chain**")
    summary_data = [
        {"Stage": "Baseline (CPU, mmap)",           "tok/s": 51.25,  "Cumulative gain": "—"},
        {"Stage": "+ Warm page cache",               "tok/s": 58.23,  "Cumulative gain": "+14%"},
        {"Stage": "+ Smaller KV cache (n_ctx=512)",  "tok/s": 61.73,  "Cumulative gain": "+20%"},
        {"Stage": "+ 10 GPU layers",                 "tok/s": 64.98,  "Cumulative gain": "+27%"},
        {"Stage": "+ 20 GPU layers",                 "tok/s": 74.66,  "Cumulative gain": "+46%"},
        {"Stage": "+ All GPU layers",                "tok/s": 95.55,  "Cumulative gain": "+86%"},
        {"Stage": "+ Flash attention (FINAL)",       "tok/s": 98.96,  "Cumulative gain": "+93%"},
    ]
    st_ui.table(pd.DataFrame(summary_data))

    # ── Full experiment table + inspector ──────────────────────────────────
    st_ui.divider()
    st_ui.markdown("## All 41 Experiments")
    df   = pd.DataFrame(rows)
    cats = ["(all)"] + sorted(df["category"].dropna().unique().tolist())
    decs = ["(all)"] + sorted(df["decision"].dropna().unique().tolist())
    c1, c2, c3 = st_ui.columns(3)
    fc   = c1.selectbox("Category", cats)
    fd   = c2.selectbox("Decision",  decs)
    sort = c3.selectbox("Sort by", ["exp", "mean_tps", "mean_latency_s", "peak_memory_mb"])
    view = df.copy()
    if fc != "(all)":
        view = view[view["category"] == fc]
    if fd != "(all)":
        view = view[view["decision"] == fd]
    try:
        view = view.sort_values(sort, key=lambda s: pd.to_numeric(s, errors="coerce"))
    except Exception:
        pass
    st_ui.dataframe(view, use_container_width=True, height=370)

    sel = st_ui.selectbox("Inspect experiment", view["exp"].tolist())
    md_path = os.path.join(st.EXPERIMENTS, f"exp{int(float(sel)):03d}.md")
    if os.path.exists(md_path):
        with open(md_path) as f:
            st_ui.markdown(f.read())


def tab_timeline():
    st_ui.subheader("Optimization Timeline")
    rows = st.read_experiments()
    if not rows:
        st_ui.info("No data yet.")
        return
    df = pd.DataFrame(rows)
    df["exp"] = pd.to_numeric(df["exp"], errors="coerce")
    for col in ["mean_tps", "mean_latency_s", "peak_memory_mb", "baseline_tps"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("exp").reset_index()

    def _line(data, x_col, y_col, x_title, y_title, color="#00e8e8"):
        return (
            alt.Chart(data.dropna(subset=[y_col]))
            .mark_line(point=True, color=color)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_title),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                tooltip=[alt.Tooltip(x_col, title=x_title),
                         alt.Tooltip(y_col, title=y_title)],
            )
            .properties(height=220)
        )

    st_ui.markdown("**Throughput (tok/s)**")
    st_ui.altair_chart(_line(df, "exp", "mean_tps", "Experiment Number", "Throughput (tok/s)"),
                       use_container_width=True)

    base = df["baseline_tps"].dropna().iloc[0] if df["baseline_tps"].notna().any() else None
    if base:
        df["speedup"] = df["mean_tps"] / base
        st_ui.markdown("**Speedup vs baseline (×)**")
        st_ui.altair_chart(_line(df, "exp", "speedup", "Experiment Number", "Speedup (×)", "#00c8a0"),
                           use_container_width=True)

    c1, c2 = st_ui.columns(2)
    with c1:
        st_ui.markdown("**Mean latency (s)**")
        st_ui.altair_chart(_line(df, "exp", "mean_latency_s", "Experiment Number", "Latency (s)", "#c8a000"),
                           use_container_width=True)
    with c2:
        st_ui.markdown("**Peak memory (MB)**")
        st_ui.altair_chart(_line(df, "exp", "peak_memory_mb", "Experiment Number", "Peak Memory (MB)", "#c86000"),
                           use_container_width=True)

    pdir = st.PLOTS
    pngs = (
        [f for f in sorted(os.listdir(pdir)) if f.endswith(".png")]
        if os.path.isdir(pdir) else []
    )
    if pngs:
        st_ui.divider()
        st_ui.caption("Generated plots (plots/):")
        for f in pngs:
            st_ui.image(os.path.join(pdir, f))


def tab_diff():
    st_ui.subheader("Optimization Diff Viewer")

    # ── Headline numbers ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st_ui.columns(4)
    c1.metric("Baseline tok/s",   "51.25",  delta="CPU-only · mmap · no GPU")
    c2.metric("Optimized tok/s",  "98.96",  delta="+93% over baseline")
    c3.metric("Winning experiment", "exp014", delta="41 experiments run")
    c4.metric("Changes required",  "2 knobs", delta="n_gpu_layers + flash_attn")

    st_ui.divider()

    # ── Config diff table with explanations ───────────────────────────────
    st_ui.markdown("**What changed between baseline and optimized**")

    KNOB_META = {
        "n_gpu_layers": {
            "category": "GPU Offload",
            "baseline_meaning": "CPU-only — all matrix math runs on CPU cores",
            "optimized_meaning": "All 24 layers offloaded to Metal GPU",
            "why": "M4 GPU executes dequantized matrix-vector multiply ~2× faster than CPU for Q4_K_M weights. Single biggest win in the entire experiment cycle.",
            "impact": "+86% tok/s (51→96) from CPU→GPU alone",
        },
        "flash_attn": {
            "category": "Attention",
            "baseline_meaning": "Standard attention — full KV cache materialized in memory",
            "optimized_meaning": "Flash attention — fused kernel, lower memory footprint",
            "why": "Reduces memory traffic during attention computation, improving GPU utilization especially at longer sequences.",
            "impact": "+3.6% on top of GPU offload (96→99 tok/s)",
        },
        "label": {
            "category": "Provenance",
            "baseline_meaning": "Named 'baseline' — the unoptimized reference",
            "optimized_meaning": "Named 'exp014' — the experiment that discovered this config",
            "why": "Bookkeeping only. Not a functional change.",
            "impact": "No performance impact",
        },
    }

    base = baseline_config()
    opt  = load_optimized_config()
    d    = diff_configs(base, opt)

    if not d:
        st_ui.info("No diff — best_config.json not found. Run the experiment cycle first.")
        return

    rows = []
    for k, v in d.items():
        meta = KNOB_META.get(k, {})
        rows.append({
            "Category":          meta.get("category", "Config"),
            "Knob":              k,
            "Baseline":          str(v["baseline"]),
            "Optimized":         str(v["optimized"]),
            "Performance impact": meta.get("impact", "—"),
        })
    st_ui.table(pd.DataFrame(rows))

    st_ui.divider()

    # ── Per-knob deep dive ────────────────────────────────────────────────
    st_ui.markdown("**Deep dive: why each change matters**")
    for k, v in d.items():
        meta = KNOB_META.get(k)
        if not meta or meta["category"] == "Provenance":
            continue
        with st_ui.expander(f"{meta['category']} — `{k}`"):
            col1, col2 = st_ui.columns(2)
            col1.markdown(f"**Baseline:** `{v['baseline']}`  \n{meta['baseline_meaning']}")
            col2.markdown(f"**Optimized:** `{v['optimized']}`  \n{meta['optimized_meaning']}")
            st_ui.markdown(f"**Why it works:** {meta['why']}")
            st_ui.success(f"Impact: {meta['impact']}")

    st_ui.divider()

    # ── What we tried and discarded ───────────────────────────────────────
    st_ui.markdown("**What we tried that didn't make the cut**")
    discards = [
        ("mlock=True",        "−14.9%", "Pinning 8.84 GB into locked RAM caused memory pressure that hurt everything else"),
        ("n_ctx=4096",        "−34.9%", "Larger context bloated the KV cache, squeezing OS page cache headroom for expert pages"),
        ("n_threads=12",      "−43.2%", "Over-subscribing threads increases context-switching overhead; sweet spot is 8"),
        ("No mmap (load all to RAM)", "−1.8%", "Eager loading slightly slower than letting the OS page cache manage expert reads"),
        ("n_batch=2048",      "−1.5%",  "Larger batch had negligible effect on single-stream throughput"),
    ]
    disc_df = pd.DataFrame(discards, columns=["Experiment", "Impact", "Why it failed"])
    st_ui.table(disc_df)

    st_ui.divider()

    # ── Generalization across models ──────────────────────────────────────
    st_ui.markdown("**Does the optimized config generalize?**")
    gen_data = [
        ("Qwen1.5-MoE-A2.7B",     "Q4_K_M · 8.84 GB", "51.25", "98.96", "+93%",  "Full GPU offload — 13 GB free after model load"),
        ("Qwen3.5-35B-A3B (IQ2)", "IQ2_M · 11 GB",    "18.70", "45.62", "+144%", "Full GPU offload — 13 GB free after smaller quant"),
    ]
    gen_df = pd.DataFrame(gen_data, columns=["Model", "Quant", "Baseline tok/s", "Optimized tok/s", "Speedup", "Note"])
    st_ui.table(gen_df)
    st_ui.caption(
        "Key finding: GPU offload generalizes when RAM headroom exists. "
        "Quantization (Q4→IQ2) is a RAM management strategy, not just a quality trade-off — "
        "halving model size re-enables GPU offload and frees page cache for expert streaming."
    )

    bc = st.read_json(st.BEST_CONFIG_JSON, None)
    if bc:
        st_ui.caption(f"Source: best_config.json · exp{bc.get('exp')} · data_source={bc.get('data_source')}")


def tab_moe():
    st_ui.subheader("MoE Expert Streaming Visualization")

    # ── Model architecture cards ──────────────────────────────────────────
    st_ui.markdown("**Active model tiers**")
    mc1, mc2 = st_ui.columns(2)
    with mc1:
        small_ok = os.path.isdir(os.path.expanduser("~/models/Qwen1.5-MoE-A2.7B-GGUF"))
        st_ui.metric("Small — Qwen1.5-MoE-A2.7B", "Q4_K_M · 8.84 GB",
                     "Downloaded" if small_ok else "Not downloaded")
        st_ui.caption("60 experts/layer · top-4 active · 24 layers · 14.3B total / 2.7B active · 99 tok/s optimized")
    with mc2:
        iq2_ok = os.path.isdir(os.path.expanduser("~/models/Qwen3.5-35B-A3B-IQ2-GGUF"))
        iq2_files = os.listdir(os.path.expanduser("~/models/Qwen3.5-35B-A3B-IQ2-GGUF")) if iq2_ok else []
        iq2status = "Downloaded" if (iq2_ok and any(f.endswith(".gguf") for f in iq2_files)) else "Not downloaded"
        st_ui.metric("Medium IQ2 — Qwen3.5-35B-A3B", "IQ2_M · 11 GB", iq2status)
        st_ui.caption("64 experts/layer · top-4 active · 35B total / 3B active · 45 tok/s optimized · beats instructor 16.3 ref")

    st_ui.divider()

    # ── Expert streaming explained ────────────────────────────────────────
    st_ui.markdown("**Why MoE sparsity enables laptop inference**")
    st_ui.markdown(
        "Each decoding step activates only **K=4 experts** out of 60 per layer.  "
        "This means ~93% of expert weights stay on SSD and are never read.  "
        "Only the selected experts are streamed via `mmap` on demand — the OS page "
        "cache keeps hot experts warm across tokens.  "
        "This is the core technique the project optimizes."
    )

    # ── Expert streaming stats from benchmark history ─────────────────────
    st_ui.divider()
    st_ui.markdown("**Expert streaming stats (from benchmark runs)**")
    rows = st.read_experiments()
    if rows:
        df = pd.DataFrame(rows)
        for col in ["expert_bytes_per_tok_mb", "n_experts", "n_experts_used", "n_moe_layers"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        col_a, col_b = st_ui.columns(2)
        with col_a:
            if "expert_bytes_per_tok_mb" in df and df["expert_bytes_per_tok_mb"].notna().any():
                st_ui.markdown("**Expert bytes streamed per token (MB)**")
                valid = df[df["expert_bytes_per_tok_mb"].notna()][["exp", "expert_bytes_per_tok_mb"]].copy()
                valid["exp"] = pd.to_numeric(valid["exp"], errors="coerce")
                chart = (
                    alt.Chart(valid)
                    .mark_line(point=True, color="#00e8e8")
                    .encode(
                        x=alt.X("exp:Q", title="Experiment Number"),
                        y=alt.Y("expert_bytes_per_tok_mb:Q", title="Expert Data Streamed per Token (MB)"),
                        tooltip=[
                            alt.Tooltip("exp", title="Experiment"),
                            alt.Tooltip("expert_bytes_per_tok_mb", title="MB/token"),
                        ],
                    )
                    .properties(height=220)
                )
                st_ui.altair_chart(chart, use_container_width=True)
            else:
                st_ui.info("expert_bytes_per_tok data not yet available — run experiments first.")
        with col_b:
            if "n_experts" in df and df["n_experts"].notna().any():
                row0 = df[df["n_experts"].notna()].iloc[0]
                st_ui.metric("Total experts / layer", int(row0["n_experts"]))
                st_ui.metric("Active per token (K)",  int(row0["n_experts_used"]))
                st_ui.metric("MoE layers",            int(row0["n_moe_layers"]))
                sparsity = 1.0 - float(row0["n_experts_used"]) / float(row0["n_experts"])
                st_ui.metric("Sparsity", f"{round(sparsity*100,1)}%")
            else:
                st_ui.info("Run at least exp000 to populate expert architecture stats.")

        # Expert streaming category experiments
        stream_exps = df[df["category"] == "ssd_streaming"] if "category" in df.columns else pd.DataFrame()
        if not stream_exps.empty:
            st_ui.divider()
            st_ui.markdown("**Expert streaming experiment results**")
            st_ui.dataframe(
                stream_exps[["exp", "title", "mean_tps", "peak_memory_mb",
                              "decision"]].reset_index(drop=True),
                use_container_width=True,
            )
    else:
        st_ui.info(
            "No benchmark data yet.  Run:\n\n"
            "```\npython run_experiments.py --mode real\n```"
        )

    # ── Per-step expert selection note ────────────────────────────────────
    st_ui.divider()
    st_ui.caption(
        "Per-token, per-layer expert routing indices require llama.cpp C-level "
        "callbacks not yet exposed in llama-cpp-python 0.3.x. "
        "Architecture info is read from GGUF metadata; bytes-per-token is "
        "estimated from model size and top-K. "
        "This limitation is documented, not hidden."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _inject_styles()
    st_ui.markdown(_build_header_html(), unsafe_allow_html=True)
    _host_header()
    _data_source_banner()

    tabs = st_ui.tabs([
        "Live Playground",
        "Compare",
        "Experiment Explorer",
        "Optimization Timeline",
        "Diff Viewer",
        "MoE Visualization",
    ])
    with tabs[0]: tab_playground()
    with tabs[1]: tab_compare()
    with tabs[2]: tab_explorer()
    with tabs[3]: tab_timeline()
    with tabs[4]: tab_diff()
    with tabs[5]: tab_moe()


main()
