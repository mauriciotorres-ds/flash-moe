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

Real generation requires torch + transformers + the model. When those aren't
present (or when results are SAMPLE_SYNTHETIC), the dashboard still renders and
shows a clear banner — it never presents synthetic data as measured.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st_ui
import pandas as pd

from ireng import storage as st
from ireng.hardware import detect_host, TARGET_SPEC
from ireng.config import baseline_config, diff_configs
from ireng.prompts import SUITE, CATEGORIES

st_ui.set_page_config(page_title="Inference Research Dashboard", layout="wide")

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen3.5-397B-A17B"]
ENGINES = ["Baseline Engine", "Optimized Engine", "Compare Both"]


# ----------------------------------------------------------------- helpers
def _data_source_banner():
    rows = st.read_experiments()
    srcs = {r.get("data_source") for r in rows}
    if "SAMPLE_SYNTHETIC" in srcs:
        st_ui.error("⚠️  SAMPLE DATA loaded — these numbers are SYNTHETIC, not "
                    "real measurements. Run `python run_experiments.py --mode real "
                    "--device mps` on your Mac to replace with measured results.")
    elif "MOCK_DRYRUN" in srcs:
        st_ui.warning("Mock dry-run data loaded (synthetic). For real numbers run "
                      "the runner in --mode real on your Mac.")
    elif rows:
        st_ui.success("Measured data loaded.")
    else:
        st_ui.info("No experiment data yet. Seed a demo with "
                   "`python seed_sample_data.py` or run the real benchmarks.")


def _host_header():
    host = detect_host()
    c1, c2, c3, c4 = st_ui.columns(4)
    c1.metric("Target", TARGET_SPEC["label"])
    c2.metric("Host device", host.device)
    c3.metric("torch", host.torch_version or "not installed")
    c4.metric("Experiments on disk", len(st.read_experiments()))


@st_ui.cache_resource(show_spinner=False)
def _get_engine(kind: str, model_id: str, device: str):
    """Lazy-load an engine. Returns (engine, error_str)."""
    try:
        if kind == "Baseline Engine":
            from baseline_engine import BaselineEngine
            return BaselineEngine(model_id, device), None
        else:
            from optimized_engine import OptimizedEngine
            return OptimizedEngine(model_id, device), None
    except Exception as e:
        return None, str(e)


# ------------------------------------------------------------------- tabs
def tab_playground():
    st_ui.subheader("Live Inference Playground")
    col_a, col_b = st_ui.columns([2, 1])
    with col_b:
        model_id = st_ui.selectbox("Model", MODELS)
        engine_kind = st_ui.selectbox("Engine", ENGINES[:2])  # single engine here
        device = st_ui.selectbox("Device", ["auto", "mps", "cuda", "cpu"])
        max_new = st_ui.slider("Max new tokens", 8, 512, 128, 8)
        temperature = st_ui.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        top_p = st_ui.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
        top_k = st_ui.slider("Top-k", 0, 200, 50, 1)
        seed = st_ui.number_input("Seed", value=1234, step=1)
        do_log = st_ui.checkbox("Log this run to dashboard_logs/", value=True)
    with col_a:
        system = st_ui.text_area("System prompt", "You are a concise, helpful assistant.", height=70)
        prompt = st_ui.text_area("Prompt", "Explain Mixture-of-Experts in two sentences.", height=120)
        run = st_ui.button("▶ Run inference", type="primary")

    if model_id == "Qwen3.5-397B-A17B":
        st_ui.info("The 397B MoE runs through the Flash-MoE Metal engine, not the "
                   "Python HF path. Use `../metal_infer/infer` / "
                   "`large_model/validate_transfer.py` for live 397B generation.")
        return
    if not run:
        return

    engine, err = _get_engine(engine_kind, model_id, device)
    if err or engine is None:
        st_ui.error(f"Engine unavailable: {err}\n\nInstall deps and download the "
                    f"model (`pip install -r requirements.txt`), then retry.")
        return

    from ireng.prompts import Prompt
    engine.config.max_new_tokens = max_new
    engine.config.do_sample = temperature > 0
    engine.config.temperature = temperature
    engine.config.top_p = top_p
    engine.config.top_k = top_k
    engine.config.seed = int(seed)
    p = Prompt("playground", "factual", system, prompt, max_new)

    out_box = st_ui.empty()
    m1, m2, m3, m4, m5 = st_ui.columns(5)
    text = ""
    last = None
    with st_ui.spinner("Generating..."):
        for chunk, m in engine.stream(p):
            if chunk:
                text += chunk
                out_box.markdown(f"```\n{text}\n```")
            last = m
            m1.metric("tokens", m.tokens_generated)
            m2.metric("tok/s", m.tokens_per_second)
            m3.metric("TTFT (s)", m.time_to_first_token_s or "—")
            m4.metric("elapsed (s)", m.total_latency_s)
            m5.metric("context", m.context_length)
    if last:
        st_ui.divider()
        d = last.as_dict()
        g = d["gpu_utilization_pct"]
        cols = st_ui.columns(5)
        cols[0].metric("peak mem (MB)", d["peak_memory_mb"])
        cols[1].metric("cur mem (MB)", d["current_memory_mb"])
        cols[2].metric("CPU %", d["cpu_utilization_pct"])
        cols[3].metric("GPU %", g if g is not None else "n/a (MPS)")
        cols[4].metric("KV cache (MB)", d["kv_cache_mb"])
        if do_log:
            _log_run(model_id, engine_kind, system, prompt, d)
            st_ui.caption("Logged to dashboard_logs/")


def _log_run(model_id, engine_kind, system, prompt, metrics):
    os.makedirs(st.LOGS, exist_ok=True)
    fn = os.path.join(st.LOGS, time.strftime("run_%Y%m%d_%H%M%S.json"))
    with open(fn, "w") as f:
        json.dump({"model": model_id, "engine": engine_kind, "system": system,
                   "prompt": prompt, "metrics": metrics,
                   "timestamp": st.now_iso()}, f, indent=2)


def tab_compare():
    st_ui.subheader("Baseline vs Optimized")
    cmp = st.read_json(os.path.join(st.RESULTS, "last_comparison.json"), None)
    if cmp:
        c1, c2, c3 = st_ui.columns(3)
        c1.metric("baseline tok/s", cmp.get("baseline_tps"))
        c2.metric("optimized tok/s", cmp.get("optimized_tps"))
        c3.metric("speedup", f"{cmp.get('speedup')}x")
        st_ui.caption(f"From benchmark_runner.py — data_source={cmp.get('data_source')}, "
                      f"device={cmp.get('device')}")
    else:
        st_ui.info("No comparison yet. Run `python benchmark_runner.py --engine both` "
                   "on your Mac.")
    st_ui.markdown("**Run a live comparison** by launching `benchmark_runner.py "
                   "--engine both`; results appear here.")


def tab_explorer():
    st_ui.subheader("Experiment Explorer")
    rows = st.read_experiments()
    if not rows:
        st_ui.info("No experiments. Seed sample data or run the runner.")
        return
    df = pd.DataFrame(rows)
    cats = ["(all)"] + sorted(df["category"].dropna().unique().tolist())
    decs = ["(all)"] + sorted(df["decision"].dropna().unique().tolist())
    c1, c2, c3 = st_ui.columns(3)
    fc = c1.selectbox("Category", cats)
    fd = c2.selectbox("Decision", decs)
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
    st_ui.dataframe(view, use_container_width=True, height=380)

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
        st_ui.markdown("**Speedup vs baseline (x)**")
        st_ui.line_chart(df["speedup"])
    c1, c2 = st_ui.columns(2)
    with c1:
        st_ui.markdown("**Mean latency (s)**")
        st_ui.line_chart(df["mean_latency_s"])
    with c2:
        st_ui.markdown("**Peak memory (MB)**")
        st_ui.line_chart(df["peak_memory_mb"])
    # static PNGs if generated
    pdir = st.PLOTS
    pngs = [f for f in sorted(os.listdir(pdir)) if f.endswith(".png")] if os.path.isdir(pdir) else []
    if pngs:
        st_ui.divider()
        st_ui.caption("Generated plots (plots/):")
        for f in pngs:
            st_ui.image(os.path.join(pdir, f))


def tab_diff():
    st_ui.subheader("Optimization Diff Viewer")
    from optimized_engine import load_optimized_config
    base = baseline_config()
    opt = load_optimized_config()
    d = diff_configs(base, opt)
    st_ui.caption(f"Optimized config label: {opt.label}")
    if not d:
        st_ui.info("Optimized config equals baseline (no best_config.json yet — "
                   "showing fallback default).")
    rows = [{"knob": k, "baseline": v["baseline"], "optimized": v["optimized"]}
            for k, v in d.items()]
    if rows:
        st_ui.table(pd.DataFrame(rows))
    bc = st.read_json(st.BEST_CONFIG_JSON, None)
    if bc:
        st_ui.caption(f"Source: best_config.json (exp{bc.get('exp')}, "
                      f"data_source={bc.get('data_source')})")


def tab_moe():
    st_ui.subheader("MoE Visualization — Qwen3.5-397B-A17B")
    transfer = st.read_json(os.path.join(st.RESULTS, "large_model_transfer.json"), None)
    st_ui.markdown(
        "The 397B model has **512 experts/layer, K=4 active per token**. Live "
        "per-expert routing counters require instrumenting the Metal engine "
        "(`metal_infer/infer.m`) to dump routing decisions.")
    if transfer:
        st_ui.json({k: transfer[k] for k in transfer if k != "transfer_map"})
        if transfer.get("transfer_map"):
            st_ui.markdown("**Optimization transfer map (small → MoE):**")
            st_ui.table(pd.DataFrame(transfer["transfer_map"]))
    else:
        st_ui.warning("Limitation: no expert-routing data captured yet. Run "
                      "`python large_model/validate_transfer.py` on your Mac after "
                      "building `metal_infer`. Expert heatmaps will populate once "
                      "the engine emits per-expert activation counts.")


def main():
    st_ui.title("🔬 Inference-Engine Research Dashboard")
    _host_header()
    _data_source_banner()
    tabs = st_ui.tabs(["Live Playground", "Compare", "Experiment Explorer",
                       "Optimization Timeline", "Diff Viewer", "MoE Visualization"])
    with tabs[0]:
        tab_playground()
    with tabs[1]:
        tab_compare()
    with tabs[2]:
        tab_explorer()
    with tabs[3]:
        tab_timeline()
    with tabs[4]:
        tab_diff()
    with tabs[5]:
        tab_moe()


main()
