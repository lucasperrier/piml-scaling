from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_ROOT = BASE_DIR / "runs"
PROGRESS_DIR = BASE_DIR / "runs-progress"
AGGREGATE_PATH = PROGRESS_DIR / "runs_aggregate.csv"
GROUPED_PATH = PROGRESS_DIR / "grouped_metrics.csv"
SCALING_PATH = PROGRESS_DIR / "scaling_fits.json"


st.set_page_config(page_title="Scaling-PIML Live Dashboard", layout="wide")


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


st.title("Scaling-PIML Live Dashboard")
st.caption("Auto-updating dashboard for sweep progress, scaling fits, and model comparisons.")

with st.sidebar:
    st.header("Refresh")
    refresh_seconds = st.slider("Auto-refresh interval (seconds)", min_value=5, max_value=120, value=20)
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_refresh")
    else:
        st.warning("Install streamlit-autorefresh for automatic updates.")

    st.header("Data sources")
    st.write(f"runs root: {RUNS_ROOT}")
    st.write(f"progress dir: {PROGRESS_DIR}")

    st.header("View")
    dashboard_mode = st.radio("Layout", options=["Paper Mode", "Standard"], index=0)

aggregate_df, grouped_df, fits = _load_data()
completed_runs = _count_completed_runs()
total_runs = 720
completion_rate = 100.0 * completed_runs / total_runs if total_runs else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Completed runs", f"{completed_runs}/{total_runs}")
c2.metric("Completion", f"{completion_rate:.1f}%")
c3.metric("Aggregate last update", _last_modified(AGGREGATE_PATH))
c4.metric("Fits last update", _last_modified(SCALING_PATH))

if aggregate_df.empty or grouped_df.empty:
    st.error(
        "No aggregated data found yet. Run aggregate script first: "
        "python scripts/aggregate_runs.py --runs-root runs --out-dir runs-progress"
    )
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
