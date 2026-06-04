"""Disk persistence — the spec's "what must always be on disk" contract.

Every experiment writes results immediately so the project survives any
interruption. This module centralises all paths and read/write helpers so the
runner, dashboard, and plot generator all agree on the on-disk format.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Project root = parent of this package's parent (ireng/ -> project/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EXPERIMENTS = os.path.join(ROOT, "experiments")
REPORTS = os.path.join(ROOT, "reports")
PLOTS = os.path.join(ROOT, "plots")
LOGS = os.path.join(ROOT, "dashboard_logs")

STATE_FILE = os.path.join(ROOT, "state.json")
FAILURE_LOG = os.path.join(ROOT, "failure_log.md")

EXPERIMENTS_CSV = os.path.join(RESULTS, "experiments.csv")
EXPERIMENTS_JSON = os.path.join(RESULTS, "experiments.json")
LEADERBOARD_CSV = os.path.join(RESULTS, "leaderboard.csv")
BENCH_HISTORY_CSV = os.path.join(RESULTS, "benchmark_history.csv")
BEST_CONFIG_JSON = os.path.join(RESULTS, "best_config.json")

for _d in (RESULTS, EXPERIMENTS, REPORTS, PLOTS, LOGS):
    os.makedirs(_d, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------- generic
def read_json(path: str, default=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return default


def write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------- state
def read_state() -> Dict[str, Any]:
    return read_json(STATE_FILE, default={
        "phase": 1, "current_experiment": 0, "last_completed_experiment": -1,
        "baseline_tps": None, "best_tps": None, "best_config": None,
        "status": "idle", "last_updated": now_iso(),
    })


def write_state(state: Dict[str, Any]):
    state["last_updated"] = now_iso()
    write_json(STATE_FILE, state)


# --------------------------------------------------------------- experiments
EXP_FIELDS = [
    "exp", "title", "category", "label", "decision",
    "mean_tps", "baseline_tps", "best_tps", "delta_pct_vs_best",
    "mean_ttft_s", "mean_latency_s", "peak_memory_mb", "kv_cache_mb",
    "quality_ok", "measured", "data_source", "device", "timestamp", "notes",
]


def _empty(path: str) -> bool:
    return (not os.path.exists(path)) or os.path.getsize(path) == 0


def append_experiment_row(row: Dict[str, Any]):
    """Append to experiments.csv and rebuild experiments.json."""
    need_header = _empty(EXPERIMENTS_CSV)
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXP_FIELDS, extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerow(row)
    _rebuild_experiments_json()


def _rebuild_experiments_json():
    rows = read_experiments()
    write_json(EXPERIMENTS_JSON, rows)


def read_experiments() -> List[Dict[str, Any]]:
    if not os.path.exists(EXPERIMENTS_CSV):
        return []
    with open(EXPERIMENTS_CSV) as f:
        return list(csv.DictReader(f))


def append_benchmark_history(rows: List[Dict[str, Any]]):
    fields = [
        "exp", "label", "prompt_id", "category", "tokens_generated",
        "tokens_per_second", "decode_tokens_per_second", "time_to_first_token_s",
        "total_latency_s", "peak_memory_mb", "current_memory_mb",
        "cpu_utilization_pct", "gpu_utilization_pct", "kv_cache_mb",
        "context_length", "device", "measured", "data_source", "timestamp",
    ]
    need_header = _empty(BENCH_HISTORY_CSV)
    with open(BENCH_HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if need_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def write_leaderboard(rows: List[Dict[str, Any]]):
    """rows already sorted best-first."""
    fields = ["rank", "exp", "label", "mean_tps", "delta_pct_vs_baseline",
              "decision", "data_source"]
    with open(LEADERBOARD_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows, 1):
            r = dict(r)
            r["rank"] = i
            w.writerow(r)


def rebuild_leaderboard():
    rows = [r for r in read_experiments() if r.get("decision") == "keep"]
    base = None
    for r in read_experiments():
        if r.get("exp") in ("0", "000", 0):
            base = _f(r.get("mean_tps"))
    def _key(r):
        return _f(r.get("mean_tps")) or 0.0
    rows.sort(key=_key, reverse=True)
    for r in rows:
        tps = _f(r.get("mean_tps"))
        if base and tps:
            r["delta_pct_vs_baseline"] = round((tps - base) / base * 100, 2)
    write_leaderboard(rows)


def write_best_config(exp: int, label: str, config_dict: Dict[str, Any],
                      mean_tps: float, data_source: str):
    write_json(BEST_CONFIG_JSON, {
        "exp": exp, "label": label, "mean_tps": mean_tps,
        "data_source": data_source, "updated": now_iso(),
        "config": config_dict,
    })


# ------------------------------------------------------------- failure log
def append_failure(exp: int, title: str, why_attempted: str, why_failed: str,
                   perf_impact: str, lessons: str):
    header_needed = _empty(FAILURE_LOG)
    with open(FAILURE_LOG, "a") as f:
        if header_needed:
            f.write("# Failure Log\n\n"
                    "Discarded experiments and why. Failures are data.\n\n")
        f.write(f"## exp{exp:03d} — {title}\n\n"
                f"- **Why attempted:** {why_attempted}\n"
                f"- **Why it failed / was discarded:** {why_failed}\n"
                f"- **Performance impact:** {perf_impact}\n"
                f"- **Lessons learned:** {lessons}\n\n")


# ------------------------------------------------------------------- expNNN
def write_experiment_md(exp: int, content: str):
    path = os.path.join(EXPERIMENTS, f"exp{exp:03d}.md")
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------- git
def git_commit(message: str) -> bool:
    try:
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True,
                       capture_output=True, timeout=30)
        r = subprocess.run(["git", "commit", "-m", message], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def git_tag(tag: str) -> bool:
    try:
        r = subprocess.run(["git", "tag", "-f", tag], cwd=ROOT,
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
