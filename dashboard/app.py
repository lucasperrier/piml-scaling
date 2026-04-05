from __future__ import annotations

import json
import os as _os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


BASE_DIR = Path(__file__).resolve().parents[1]

_runs_name = _os.environ.get("DASHBOARD_RUNS", "runs-dense")
RUNS_ROOT = BASE_DIR / _runs_name
PROGRESS_DIR = RUNS_ROOT
AGGREGATE_PATH = PROGRESS_DIR / "runs_aggregate.csv"
GROUPED_PATH = PROGRESS_DIR / "grouped_metrics.csv"
SCALING_PATH = PROGRESS_DIR / "scaling_fits.json"
LOGS_DIR = BASE_DIR / "logs"

MODELS = ["plain", "piml", "piml-simpson"]
TARGET_PER_MODEL = 693
TARGET_TOTAL = TARGET_PER_MODEL * len(MODELS)

# Remote pod settings (SSH alias from ~/.ssh/config)
REMOTE_HOST = _os.environ.get("DASHBOARD_REMOTE_HOST", "runpod")
REMOTE_PROJECT = _os.environ.get("DASHBOARD_REMOTE_PROJECT", "/workspace/projects/piml-scaling")
REMOTE_RUNS = f"{REMOTE_PROJECT}/{_runs_name}"

st.set_page_config(page_title="Scaling-PIML Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# SSH helpers — poll the pod from your local machine
# ---------------------------------------------------------------------------

def _ssh_cmd(cmd: str, timeout: int = 10) -> str | None:
    """Run a command on the remote pod via SSH. Returns stdout or None on failure."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", REMOTE_HOST, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


@st.cache_data(ttl=8)
def _remote_model_counts() -> dict[str, int]:
    """Get per-model completed run counts from the pod."""
    out = _ssh_cmd(
        f"cd {REMOTE_PROJECT} && "
        f"for m in plain piml piml-simpson; do "
        f"  c=$(find {REMOTE_RUNS}/model=$m -name metrics.json 2>/dev/null | wc -l); "
        f'  echo "$m $c"; '
        f"done"
    )
    counts: dict[str, int] = {}
    if out:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2:
                counts[parts[0]] = int(parts[1])
    return counts


@st.cache_data(ttl=8)
def _remote_capacity_grid() -> pd.DataFrame:
    """Get (model, capacity, completed) grid from the pod."""
    out = _ssh_cmd(
        f"cd {REMOTE_PROJECT} && "
        f"for md in $(ls -d {REMOTE_RUNS}/model=* 2>/dev/null); do "
        f"  m=$(basename $md | cut -d= -f2); "
        f"  for cd_ in $(ls -d $md/capacity=* 2>/dev/null); do "
        f"    cap=$(basename $cd_ | cut -d= -f2); "
        f"    n=$(find $cd_ -name metrics.json | wc -l); "
        f'    echo "$m $cap $n"; '
        f"  done; "
        f"done",
        timeout=15,
    )
    rows = []
    if out:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3:
                rows.append({"model": parts[0], "capacity": parts[1], "completed": int(parts[2])})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["model", "capacity", "completed"])


@st.cache_data(ttl=8)
def _remote_gpu_stats() -> dict:
    """Query nvidia-smi on the pod."""
    out = _ssh_cmd(
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits"
    )
    if out:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 6:
            return {
                "name": parts[0],
                "util_pct": float(parts[1]),
                "mem_used_mb": float(parts[2]),
                "mem_total_mb": float(parts[3]),
                "temp_c": float(parts[4]),
                "power_w": float(parts[5]),
            }
    return {}


@st.cache_data(ttl=8)
def _remote_log_tail(n: int = 40) -> tuple[str, str]:
    """Get the tail of the latest step21 log from the pod. Returns (filename, content)."""
    out = _ssh_cmd(
        f"cd {REMOTE_PROJECT} && "
        f"f=$(ls -1t logs/step21_*.log logs/step21_full_*.log 2>/dev/null | head -n 1); "
        f'[ -n "$f" ] && echo "FILE:$f" && tail -n {n} "$f" || echo "FILE:none"',
        timeout=10,
    )
    if out:
        lines = out.splitlines()
        fname = lines[0].replace("FILE:", "") if lines else "none"
        content = "\n".join(lines[1:]) if len(lines) > 1 else "(empty)"
        return fname, content
    return "unreachable", "(SSH connection failed)"


@st.cache_data(ttl=8)
def _remote_process_check() -> str:
    """Check if a sweep process is running on the pod."""
    out = _ssh_cmd("ps -eo pid,etime,args | grep run_sweep | grep -v grep | head -n 3")
    return out or "(no active sweep process)"


@st.cache_data(ttl=30)
def _remote_run_timestamps_summary() -> dict:
    """Get timestamp stats from the pod for ETA estimation."""
    out = _ssh_cmd(
        f"cd {REMOTE_PROJECT} && "
        f'.venv/bin/python -c "'
        f"import os, json; "
        f"ts = sorted(os.path.getmtime(os.path.join(r,f)) "
        f"  for r,_,fs in os.walk('{REMOTE_RUNS}') for f in fs if f=='metrics.json'); "
        f"print(json.dumps({{'count':len(ts),'first':ts[0] if ts else 0,'last':ts[-1] if ts else 0,"
        f"'recent_20':ts[-20:] if len(ts)>=20 else ts}}))"
        f'"',
        timeout=15,
    )
    if out:
        try:
            return json.loads(out)
        except Exception:
            pass
    return {}


def _estimate_eta_from_summary(completed: int, total: int, summary: dict) -> str:
    remaining = total - completed
    if remaining <= 0:
        return "Done!"
    recent = summary.get("recent_20", [])
    if len(recent) < 3:
        return "Collecting data..."
    elapsed = recent[-1] - recent[0]
    n_in_window = len(recent) - 1
    if elapsed <= 0 or n_in_window <= 0:
        return "Estimating..."
    rate = n_in_window / elapsed
    eta_s = remaining / rate
    if eta_s < 60:
        return f"~{eta_s:.0f}s"
    if eta_s < 3600:
        return f"~{eta_s / 60:.0f}m"
    return f"~{eta_s / 3600:.1f}h"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _count_completed_runs() -> int:
    if not RUNS_ROOT.exists():
        return 0
    return sum(1 for _ in RUNS_ROOT.rglob("metrics.json"))


def _last_modified(path: Path) -> str:
    if not path.exists():
        return "n/a"
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aggregate = _safe_read_csv(AGGREGATE_PATH)
    grouped = _safe_read_csv(GROUPED_PATH)
    fits = _safe_read_json(SCALING_PATH)
    return aggregate, grouped, fits


# ---------------------------------------------------------------------------
# Live monitor helpers
# ---------------------------------------------------------------------------

def _model_capacity_progress() -> pd.DataFrame:
    """Build a DataFrame of (model, capacity, completed, total) from the runs tree."""
    rows = []
    if not RUNS_ROOT.exists():
        return pd.DataFrame(rows, columns=["model", "capacity", "completed"])
    for model_dir in sorted(RUNS_ROOT.glob("model=*")):
        model = model_dir.name.split("=", 1)[1]
        for cap_dir in sorted(model_dir.glob("capacity=*")):
            cap = cap_dir.name.split("=", 1)[1]
            n = sum(1 for _ in cap_dir.rglob("metrics.json"))
            rows.append({"model": model, "capacity": cap, "completed": n})
    return pd.DataFrame(rows)


def _per_model_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not RUNS_ROOT.exists():
        return counts
    for md in sorted(RUNS_ROOT.glob("model=*")):
        mname = md.name.split("=", 1)[1]
        counts[mname] = sum(1 for _ in md.rglob("metrics.json"))
    return counts


def _latest_log_path() -> Path | None:
    if not LOGS_DIR.exists():
        return None
    logs = sorted(LOGS_DIR.glob("step21_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        logs = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _tail_log(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(unable to read log)"


def _gpu_stats() -> dict:
    """Query nvidia-smi for GPU stats. Returns empty dict if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 6:
            return {
                "name": parts[0],
                "util_pct": float(parts[1]),
                "mem_used_mb": float(parts[2]),
                "mem_total_mb": float(parts[3]),
                "temp_c": float(parts[4]),
                "power_w": float(parts[5]),
            }
    except Exception:
        pass
    return {}


def _run_timestamps() -> list[float]:
    """Collect mtime of every metrics.json under RUNS_ROOT, sorted ascending."""
    if not RUNS_ROOT.exists():
        return []
    times = []
    for p in RUNS_ROOT.rglob("metrics.json"):
        times.append(p.stat().st_mtime)
    times.sort()
    return times


def _estimate_eta(completed: int, total: int, timestamps: list[float]) -> str:
    remaining = total - completed
    if remaining <= 0:
        return "Done!"
    if len(timestamps) < 5:
        return "Collecting data..."
    # Use last 20 runs to estimate recent rate
    window = timestamps[-min(20, len(timestamps)):]
    elapsed = window[-1] - window[0]
    n_in_window = len(window) - 1
    if elapsed <= 0 or n_in_window <= 0:
        return "Estimating..."
    rate = n_in_window / elapsed  # runs per second
    eta_s = remaining / rate
    if eta_s < 60:
        return f"~{eta_s:.0f}s"
    if eta_s < 3600:
        return f"~{eta_s / 60:.0f}m"
    return f"~{eta_s / 3600:.1f}h"


st.title("Scaling-PIML Dashboard")

with st.sidebar:
    st.header("Refresh")
    refresh_seconds = st.slider("Auto-refresh interval (seconds)", min_value=5, max_value=120, value=15)
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_refresh")
    else:
        st.warning("Install streamlit-autorefresh for automatic updates.")

    st.header("Data sources")
    st.write(f"runs root: `{RUNS_ROOT}`")
    st.write(f"logs dir: `{LOGS_DIR}`")

    st.header("View")
    dashboard_mode = st.radio("Layout", options=["Live Monitor", "Paper Mode", "Standard"], index=0)

# ===== LIVE MONITOR TAB =====
if dashboard_mode == "Live Monitor":
    model_counts = _remote_model_counts()
    completed_runs = sum(model_counts.values())
    ts_summary = _remote_run_timestamps_summary()
    eta = _estimate_eta_from_summary(completed_runs, TARGET_TOTAL, ts_summary)

    # ---- Top KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Progress", f"{completed_runs} / {TARGET_TOTAL}")
    k2.metric("Completion", f"{100 * completed_runs / TARGET_TOTAL:.1f}%")
    k3.metric("ETA", eta)
    gpu = _remote_gpu_stats()
    k4.metric("GPU", f"{gpu.get('name', 'N/A')}")

    st.progress(min(completed_runs / TARGET_TOTAL, 1.0))

    # ---- GPU gauges ----
    if gpu:
        st.subheader("GPU")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Utilization", f"{gpu['util_pct']:.0f}%")
        g2.metric("VRAM", f"{gpu['mem_used_mb']:.0f} / {gpu['mem_total_mb']:.0f} MB")
        g3.metric("Temperature", f"{gpu['temp_c']:.0f} °C")
        g4.metric("Power", f"{gpu['power_w']:.0f} W")
    else:
        st.warning("GPU stats unavailable — check pod connectivity.")

    # ---- Per-model progress bars ----
    st.subheader("Per-Model Progress")
    cols = st.columns(len(MODELS))
    for i, m in enumerate(MODELS):
        c = model_counts.get(m, 0)
        pct = c / TARGET_PER_MODEL if TARGET_PER_MODEL else 0
        cols[i].metric(m, f"{c} / {TARGET_PER_MODEL}")
        cols[i].progress(min(pct, 1.0))

    # ---- Process status ----
    st.subheader("Active Process")
    st.code(_remote_process_check(), language="text")

    # ---- Capacity breakdown heatmap ----
    st.subheader("Progress Grid (model × capacity)")
    cap_df = _remote_capacity_grid()
    if not cap_df.empty:
        pivot = cap_df.pivot_table(index="model", columns="capacity", values="completed", fill_value=0)
        fig_grid = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            text=pivot.values.astype(int).astype(str),
            texttemplate="%{text}",
            colorscale="Greens",
            showscale=False,
        ))
        fig_grid.update_layout(
            title="Completed runs per cell",
            xaxis_title="Capacity",
            yaxis_title="Model",
            height=250,
        )
        st.plotly_chart(fig_grid, use_container_width=True)

    # ---- Completion timeline ----
    recent_ts = ts_summary.get("recent_20", [])
    if len(recent_ts) >= 3:
        st.subheader("Recent Throughput")
        diffs = [recent_ts[i + 1] - recent_ts[i] for i in range(len(recent_ts) - 1)]
        avg_sec = sum(diffs) / len(diffs) if diffs else 0
        r1, r2 = st.columns(2)
        r1.metric("Avg seconds/run (recent)", f"{avg_sec:.1f}s")
        r2.metric("Runs/hour (recent)", f"{3600 / avg_sec:.0f}" if avg_sec > 0 else "N/A")

    # ---- Log tail ----
    st.subheader("Live Log")
    log_name, log_content = _remote_log_tail(n=40)
    st.caption(f"`{log_name}`")
    st.code(log_content, language="text")

    st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}  •  Polling: `{REMOTE_HOST}`")
    st.stop()

aggregate_df, grouped_df, fits = _load_data()
completed_runs = _count_completed_runs()

# Live progress: count per model directly from metrics.json files
_model_counts = _per_model_counts()

st.subheader("Live Sweep Progress")
_prog_cols = st.columns(max(1, len(_model_counts) + 1))
for i, (_mn, _mc) in enumerate(_model_counts.items()):
    _prog_cols[i].metric(f"{_mn}", f"{_mc}/{TARGET_PER_MODEL}")
_prog_cols[len(_model_counts)].metric("Total", f"{completed_runs}/{TARGET_TOTAL}")
st.progress(min(completed_runs / TARGET_TOTAL, 1.0))

completion_rate = 100.0 * completed_runs / TARGET_TOTAL if TARGET_TOTAL else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Completed runs", f"{completed_runs}/{TARGET_TOTAL}")
c2.metric("Completion", f"{completion_rate:.1f}%")
c3.metric("Aggregate last update", _last_modified(AGGREGATE_PATH))
c4.metric("Fits last update", _last_modified(SCALING_PATH))

if aggregate_df.empty or grouped_df.empty:
    st.info(
        "No aggregated data found yet. Run aggregate script to see scaling plots: "
        f"`python scripts/aggregate_runs.py {RUNS_ROOT}`"
    )
    st.caption("Dashboard auto-refreshes and updates when new metrics.json files appear.")
    st.stop()

# Normalize types
for col in ["dataset_size", "parameter_count", "n_runs", "n_attempted"]:
    if col in grouped_df.columns:
        grouped_df[col] = pd.to_numeric(grouped_df[col], errors="coerce")

for col in ["test_rel_l2_mean", "test_rel_l2_std", "test_rel_l2_stderr", "divergence_rate", "nan_rate"]:
    if col in grouped_df.columns:
        grouped_df[col] = pd.to_numeric(grouped_df[col], errors="coerce")

if "hidden_widths" in grouped_df.columns:
    grouped_df["hidden_widths"] = grouped_df["hidden_widths"].astype(str)

# Filters
st.subheader("Filters")
f1, f2, f3 = st.columns(3)
models = sorted(grouped_df["model_name"].dropna().unique().tolist())
capacities = sorted(grouped_df["capacity_name"].dropna().unique().tolist())
datasets = sorted(grouped_df["dataset_size"].dropna().astype(int).unique().tolist())

selected_models = f1.multiselect("Models", models, default=models)
selected_capacities = f2.multiselect("Capacities", capacities, default=capacities)
selected_datasets = f3.multiselect("Dataset sizes", datasets, default=datasets)

view_df = grouped_df[
    grouped_df["model_name"].isin(selected_models)
    & grouped_df["capacity_name"].isin(selected_capacities)
    & grouped_df["dataset_size"].isin(selected_datasets)
].copy()

if view_df.empty:
    st.warning("No data for current filters.")
    st.stop()

full_fits = fits.get("full_fits", []) if isinstance(fits, dict) else []

if dashboard_mode == "Paper Mode":
    st.subheader("Paper Mode: Key Results")

    # 1) Decision KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Filtered grouped points", f"{len(view_df)}")
    overlap = view_df.pivot_table(
        index=["capacity_name", "dataset_size"],
        columns="model_name",
        values="test_rel_l2_mean",
    ).dropna()
    k2.metric("Matched overlap points", f"{len(overlap)}")
    mean_gain = float("nan")
    if len(overlap):
        overlap["piml_gain_vs_plain"] = (overlap["plain"] - overlap["piml"]) / overlap["plain"]
        mean_gain = float(overlap["piml_gain_vs_plain"].mean())
    k3.metric("Mean piml gain vs plain", f"{mean_gain:.3f}" if pd.notna(mean_gain) else "n/a")

    # 2) Core scaling plots
    s1, s2 = st.columns(2)
    fig_data = px.line(
        view_df.sort_values("dataset_size"),
        x="dataset_size",
        y="test_rel_l2_mean",
        color="model_name",
        line_dash="capacity_name",
        markers=True,
        error_y="test_rel_l2_stderr" if "test_rel_l2_stderr" in view_df.columns else None,
        title="Error vs dataset size (D)",
    )
    fig_data.update_xaxes(type="log")
    fig_data.update_yaxes(type="log")
    s1.plotly_chart(fig_data, width="stretch")

    fig_capacity = px.line(
        view_df.sort_values("parameter_count"),
        x="parameter_count",
        y="test_rel_l2_mean",
        color="model_name",
        line_dash="dataset_size",
        markers=True,
        error_y="test_rel_l2_stderr" if "test_rel_l2_stderr" in view_df.columns else None,
        title="Error vs parameter count (N)",
    )
    fig_capacity.update_xaxes(type="log")
    fig_capacity.update_yaxes(type="log")
    s2.plotly_chart(fig_capacity, width="stretch")

    # 3) Exponent comparison table
    st.subheader("Full-Surface Exponents")
    if full_fits:
        rows = []
        for rec in full_fits:
            boot = rec.get("bootstrap", {})
            rows.append(
                {
                    "model": rec.get("model_name"),
                    "points": rec.get("n_points"),
                    "alpha_mean": boot.get("alpha_mean"),
                    "alpha_ci_lo": boot.get("alpha_ci_lo"),
                    "alpha_ci_hi": boot.get("alpha_ci_hi"),
                    "beta_mean": boot.get("beta_mean"),
                    "beta_ci_lo": boot.get("beta_ci_lo"),
                    "beta_ci_hi": boot.get("beta_ci_hi"),
                    "bootstrap_n": boot.get("n_boot_success"),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")

    # 4) Overlap comparison table
    st.subheader("Matched Plain vs PIML Comparison")
    if len(overlap):
        cmp_df = overlap.reset_index().sort_values("piml_gain_vs_plain", ascending=False)
        st.dataframe(cmp_df, width="stretch")
    else:
        st.info("No matched overlap points under current filters.")

else:
    # Progress and stability panels
    st.subheader("Run Health")
    left, right = st.columns(2)

    if "status" in aggregate_df.columns:
        status_counts = aggregate_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(status_counts, x="status", y="count", title="Run status distribution")
        left.plotly_chart(fig_status, width="stretch")

    stability = (
        view_df.groupby("model_name", as_index=False)[["divergence_rate", "nan_rate"]]
        .mean()
        .melt(id_vars="model_name", var_name="metric", value_name="rate")
    )
    fig_stability = px.bar(
        stability,
        x="model_name",
        y="rate",
        color="metric",
        barmode="group",
        title="Average divergence / NaN rates",
    )
    right.plotly_chart(fig_stability, width="stretch")

    # Scaling plots
    st.subheader("Scaling Trends")
    a, b = st.columns(2)

    fig_data = px.line(
        view_df.sort_values("dataset_size"),
        x="dataset_size",
        y="test_rel_l2_mean",
        color="model_name",
        line_dash="capacity_name",
        markers=True,
        error_y="test_rel_l2_stderr" if "test_rel_l2_stderr" in view_df.columns else None,
        title="Error vs dataset size (D)",
    )
    fig_data.update_xaxes(type="log")
    fig_data.update_yaxes(type="log")
    a.plotly_chart(fig_data, width="stretch")

    fig_capacity = px.line(
        view_df.sort_values("parameter_count"),
        x="parameter_count",
        y="test_rel_l2_mean",
        color="model_name",
        line_dash="dataset_size",
        markers=True,
        error_y="test_rel_l2_stderr" if "test_rel_l2_stderr" in view_df.columns else None,
        title="Error vs parameter count (N)",
    )
    fig_capacity.update_xaxes(type="log")
    fig_capacity.update_yaxes(type="log")
    b.plotly_chart(fig_capacity, width="stretch")

    # Heatmaps by model
    st.subheader("Error Surface")
    for model_name in selected_models:
        model_df = view_df[view_df["model_name"] == model_name]
        if model_df.empty:
            continue
        pivot = model_df.pivot_table(
            index="parameter_count",
            columns="dataset_size",
            values="test_rel_l2_mean",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        fig_heat = px.imshow(
            pivot,
            labels={"x": "dataset_size", "y": "parameter_count", "color": "test_rel_l2"},
            title=f"{model_name}: test error heatmap",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, width="stretch")

    # Exponent cards
    st.subheader("Scaling Exponent Summary")
    if full_fits:
        cards = st.columns(max(1, len(full_fits)))
        for i, rec in enumerate(full_fits):
            boot = rec.get("bootstrap", {})
            with cards[i]:
                st.markdown(f"### {rec.get('model_name', 'unknown')}")
                st.write(f"points: {rec.get('n_points', 'n/a')}")
                if boot:
                    st.write(
                        f"alpha: {boot.get('alpha_mean', float('nan')):.3f} "
                        f"[{boot.get('alpha_ci_lo', float('nan')):.3f}, {boot.get('alpha_ci_hi', float('nan')):.3f}]"
                    )
                    st.write(
                        f"beta: {boot.get('beta_mean', float('nan')):.3f} "
                        f"[{boot.get('beta_ci_lo', float('nan')):.3f}, {boot.get('beta_ci_hi', float('nan')):.3f}]"
                    )
                    st.write(f"bootstrap n: {boot.get('n_boot_success', 0)}")

    # Best runs table
    st.subheader("Top Runs (lowest test_rel_l2)")
    if "test_rel_l2" in aggregate_df.columns:
        top_cols = [
            c
            for c in [
                "model_name",
                "capacity_name",
                "dataset_size",
                "parameter_count",
                "data_seed",
                "train_seed",
                "test_rel_l2",
                "status",
            ]
            if c in aggregate_df.columns
        ]
        top = aggregate_df.sort_values("test_rel_l2", ascending=True).head(30)[top_cols]
        st.dataframe(top, width="stretch")

st.caption("Dashboard auto-refreshes and updates when new metrics.json files appear.")
