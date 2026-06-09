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

import streamlit as st_ui
import pandas as pd

from ireng import storage as st
from ireng.hardware import detect_host, TARGET_SPEC
from ireng.config import baseline_config, diff_configs
from ireng.prompts import SUITE, CATEGORIES

st_ui.set_page_config(
    page_title="Inference Research Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen3.5-397B-A17B"]
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
        + '<div class="dash-badge">Phase&#8209;1 Complete &nbsp;&#183;&nbsp; 42 Experiments</div>'
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
    c2.metric("Host device", host.device)
    c3.metric("torch", host.torch_version or "not installed")
    c4.metric("Experiments on disk", len(st.read_experiments()))


@st_ui.cache_resource(show_spinner=False)
def _get_engine(kind: str, model_id: str, device: str):
    try:
        if kind == "Baseline Engine":
            from baseline_engine import BaselineEngine
            return BaselineEngine(model_id, device), None
        else:
            from optimized_engine import OptimizedEngine
            return OptimizedEngine(model_id, device), None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────

def tab_playground():
    st_ui.subheader("Live Inference Playground")
    col_a, col_b = st_ui.columns([2, 1])
    with col_b:
        model_id  = st_ui.selectbox("Model", MODELS)
        engine_kind = st_ui.selectbox("Engine", ENGINES[:2])
        device    = st_ui.selectbox("Device", ["auto", "mps", "cuda", "cpu"])
        max_new   = st_ui.slider("Max new tokens", 8, 512, 128, 8)
        temperature = st_ui.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        top_p     = st_ui.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
        top_k     = st_ui.slider("Top-k", 0, 200, 50, 1)
        seed      = st_ui.number_input("Seed", value=1234, step=1)
        do_log    = st_ui.checkbox("Log this run to dashboard_logs/", value=True)
    with col_a:
        system = st_ui.text_area("System prompt", "You are a concise, helpful assistant.", height=70)
        prompt = st_ui.text_area("Prompt", "Explain Mixture-of-Experts in two sentences.", height=120)
        run    = st_ui.button("▶  Run Inference", type="primary")

    if model_id == "Qwen3.5-397B-A17B":
        st_ui.info(
            "The 397B MoE runs through the Flash-MoE Metal engine, not the Python HF path.  "
            "Use `../metal_infer/infer` or `large_model/validate_transfer.py` for live 397B generation."
        )
        return
    if not run:
        return

    engine, err = _get_engine(engine_kind, model_id, device)
    if err or engine is None:
        st_ui.error(
            f"Engine unavailable: {err}\n\n"
            "Install deps and download the model (`pip install -r requirements.txt`), then retry."
        )
        return

    from ireng.prompts import Prompt
    engine.config.max_new_tokens = max_new
    engine.config.do_sample      = temperature > 0
    engine.config.temperature    = temperature
    engine.config.top_p          = top_p
    engine.config.top_k          = top_k
    engine.config.seed           = int(seed)
    p = Prompt("playground", "factual", system, prompt, max_new)

    out_box = st_ui.empty()
    m1, m2, m3, m4, m5 = st_ui.columns(5)
    text = ""
    last = None
    with st_ui.spinner("Generating…"):
        for chunk, m in engine.stream(p):
            if chunk:
                text += chunk
                out_box.markdown(f"```\n{text}\n```")
            last = m
            m1.metric("tokens",    m.tokens_generated)
            m2.metric("tok/s",     m.tokens_per_second)
            m3.metric("TTFT (s)",  m.time_to_first_token_s or "—")
            m4.metric("elapsed (s)", m.total_latency_s)
            m5.metric("context",   m.context_length)
    if last:
        st_ui.divider()
        d = last.as_dict()
        g = d["gpu_utilization_pct"]
        cols = st_ui.columns(6)
        cols[0].metric("total elapsed (s)", d["total_latency_s"])
        cols[1].metric("peak mem (MB)",    d["peak_memory_mb"])
        cols[2].metric("cur mem (MB)",     d["current_memory_mb"])
        cols[3].metric("CPU %",            d["cpu_utilization_pct"])
        cols[4].metric("GPU %",            g if g is not None else "n/a (MPS)")
        cols[5].metric("KV cache (MB)",    d["kv_cache_mb"])
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
    st_ui.subheader("Baseline vs Optimized")
    cmp = st.read_json(os.path.join(st.RESULTS, "last_comparison.json"), None)
    if cmp:
        c1, c2, c3 = st_ui.columns(3)
        c1.metric("Baseline tok/s",  cmp.get("baseline_tps"))
        c2.metric("Optimized tok/s", cmp.get("optimized_tps"))
        c3.metric("Speedup",         f"{cmp.get('speedup')}×")
        st_ui.caption(
            f"Source: benchmark_runner.py — "
            f"data_source={cmp.get('data_source')}  device={cmp.get('device')}"
        )
    else:
        st_ui.info("No comparison yet. Run `python benchmark_runner.py --engine both` on your Mac.")
    st_ui.markdown(
        "Re-run `benchmark_runner.py --engine both` at any time; "
        "the numbers here will update automatically."
    )


def tab_explorer():
    st_ui.subheader("Experiment Explorer")
    rows = st.read_experiments()
    if not rows:
        st_ui.info("No experiments. Seed sample data or run the runner.")
        return
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
    df = df.sort_values("exp").set_index("exp")

    st_ui.markdown("**Throughput (tok/s)**")
    st_ui.line_chart(df["mean_tps"])

    base = df["baseline_tps"].dropna().iloc[0] if df["baseline_tps"].notna().any() else None
    if base:
        df["speedup"] = df["mean_tps"] / base
        st_ui.markdown("**Speedup vs baseline (×)**")
        st_ui.line_chart(df["speedup"])

    c1, c2 = st_ui.columns(2)
    with c1:
        st_ui.markdown("**Mean latency (s)**")
        st_ui.line_chart(df["mean_latency_s"])
    with c2:
        st_ui.markdown("**Peak memory (MB)**")
        st_ui.line_chart(df["peak_memory_mb"])

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
    from optimized_engine import load_optimized_config
    base = baseline_config()
    opt  = load_optimized_config()
    d    = diff_configs(base, opt)
    st_ui.caption(f"Optimized config label: {opt.label}")
    if not d:
        st_ui.info(
            "Optimized config equals baseline "
            "(no best_config.json yet — showing fallback default)."
        )
    rows = [
        {"knob": k, "baseline": v["baseline"], "optimized": v["optimized"]}
        for k, v in d.items()
    ]
    if rows:
        st_ui.table(pd.DataFrame(rows))
    bc = st.read_json(st.BEST_CONFIG_JSON, None)
    if bc:
        st_ui.caption(
            f"Source: best_config.json  "
            f"(exp{bc.get('exp')}, data_source={bc.get('data_source')})"
        )


def tab_moe():
    st_ui.subheader("MoE Visualization — Qwen3.5-397B-A17B")
    transfer = st.read_json(os.path.join(st.RESULTS, "large_model_transfer.json"), None)
    st_ui.markdown(
        "The 397B model has **512 experts / layer, K=4 active per token**.  "
        "Live per-expert routing counters require instrumenting the Metal engine "
        "(`metal_infer/infer.m`) to emit activation counts."
    )
    if transfer:
        st_ui.json({k: transfer[k] for k in transfer if k != "transfer_map"})
        if transfer.get("transfer_map"):
            st_ui.markdown("**Optimization transfer map — small model → MoE engine:**")
            st_ui.table(pd.DataFrame(transfer["transfer_map"]))
    else:
        st_ui.warning(
            "No expert-routing data captured yet.  "
            "Build `metal_infer` and run `python large_model/validate_transfer.py`.  "
            "Expert heatmaps will populate once the engine emits per-expert activation counts."
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
