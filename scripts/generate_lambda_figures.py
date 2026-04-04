"""Generate lambda-sweep figures for midpoint and conservation priors."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
    })


CAPACITY_COLORS = {"small": "#2ca02c", "large": "#9467bd"}
DATASET_MARKERS = {128: "o", 1024: "s", 4096: "D"}


def fig_lambda_sweep_midpoint(df: pd.DataFrame, out_dir: Path) -> None:
    """Lambda sweep for midpoint ODE-residual prior."""
    # Filter to lambda >= 0 (exclude plain baselines at -1)
    sweep = df[df["lambda_phys"] >= 0].copy()
    if sweep.empty:
        print("  No midpoint lambda sweep data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, cap in zip(axes, ["small", "large"]):
        sub = sweep[sweep["capacity_name"] == cap]
        for D in sorted(sub["dataset_size"].unique()):
            row = sub[sub["dataset_size"] == D].sort_values("lambda_phys")
            if row.empty:
                continue
            lam = row["lambda_phys"].values
            E = row["test_rel_l2_mean"].values
            E_err = row["test_rel_l2_stderr"].values
            # Shift lambda=0 slightly for log scale
            lam_plot = np.where(lam == 0, 5e-5, lam)
            ax.errorbar(lam_plot, E, yerr=E_err,
                        marker=DATASET_MARKERS.get(D, "^"),
                        capsize=3, label=f"$D={D}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda_{\\mathrm{phys}}$")
        ax.set_ylabel("Test relative $\\ell_2$")
        N = sweep[sweep["capacity_name"] == cap]["parameter_count"].iloc[0] if len(sweep[sweep["capacity_name"] == cap]) > 0 else "?"
        ax.set_title(f"{cap.capitalize()} ($N = {N:,}$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.015, color="gray", ls=":", alpha=0.5, label="Physics floor")

    fig.suptitle("Midpoint-Residual Prior: $\\lambda$ Sensitivity", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lambda_midpoint.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_lambda_midpoint.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_lambda_midpoint")


def fig_lambda_sweep_conservation(df: pd.DataFrame, out_dir: Path) -> None:
    """Lambda sweep for conservation-law prior."""
    sweep = df[df["lambda_phys"] >= 0].copy()
    if sweep.empty:
        print("  No conservation lambda sweep data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, cap in zip(axes, ["small", "large"]):
        sub = sweep[sweep["capacity_name"] == cap]
        for D in sorted(sub["dataset_size"].unique()):
            row = sub[sub["dataset_size"] == D].sort_values("lambda_phys")
            if row.empty:
                continue
            lam = row["lambda_phys"].values
            E = row["test_rel_l2_mean"].values
            E_err = row["test_rel_l2_stderr"].values
            lam_plot = np.where(lam == 0, 5e-5, lam)
            ax.errorbar(lam_plot, E, yerr=E_err,
                        marker=DATASET_MARKERS.get(D, "^"),
                        capsize=3, label=f"$D={D}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda_{\\mathrm{phys}}$")
        ax.set_ylabel("Test relative $\\ell_2$")
        N = sweep[sweep["capacity_name"] == cap]["parameter_count"].iloc[0] if len(sweep[sweep["capacity_name"] == cap]) > 0 else "?"
        ax.set_title(f"{cap.capitalize()} ($N = {N:,}$)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Conservation-Law Prior: $\\lambda$ Sensitivity", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lambda_conservation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_lambda_conservation.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_lambda_conservation")


def fig_comparative_data_scaling(grouped_df: pd.DataFrame, out_dir: Path) -> None:
    """Plain vs midpoint PIML, test error vs D, at large capacity."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    MODEL_COLORS = {"plain": "#1f77b4", "piml": "#d62728"}
    MODEL_LABELS = {"plain": "Plain MLP", "piml": "PIML (midpoint)"}
    MARKERS = {"plain": "o", "piml": "s"}

    for model in ["plain", "piml"]:
        sub = grouped_df[
            (grouped_df["model_name"] == model) &
            (grouped_df["capacity_name"] == "large")
        ].sort_values("dataset_size")
        if sub.empty:
            continue
        D = sub["dataset_size"].values
        E = sub["test_rel_l2_mean"].values
        E_err = sub["test_rel_l2_stderr"].values
        ax.errorbar(D, E, yerr=E_err,
                    label=MODEL_LABELS.get(model, model),
                    color=MODEL_COLORS.get(model),
                    marker=MARKERS.get(model),
                    capsize=3, lw=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size $D$")
    ax.set_ylabel("Test relative $\\ell_2$")
    ax.set_title("Data Scaling: Plain vs Midpoint PIML (Large, $N = 67{,}074$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_comparative_data_scaling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_comparative_data_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_comparative_data_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midpoint-sweep", type=str, default="runs-lambda-sweep/lambda_sweep_grouped.csv")
    parser.add_argument("--conservation-sweep", type=str, default="runs-conservation-lambda-sweep/lambda_sweep_grouped.csv")
    parser.add_argument("--grouped-metrics", type=str, default="runs-progress/grouped_metrics.csv")
    parser.add_argument("--out-dir", type=str, default="figures-progress")
    args = parser.parse_args()

    _setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating lambda sweep figures...")

    if Path(args.midpoint_sweep).exists():
        midpoint_df = pd.read_csv(args.midpoint_sweep)
        fig_lambda_sweep_midpoint(midpoint_df, out_dir)

    if Path(args.conservation_sweep).exists():
        cons_df = pd.read_csv(args.conservation_sweep)
        fig_lambda_sweep_conservation(cons_df, out_dir)

    if Path(args.grouped_metrics).exists():
        grouped_df = pd.read_csv(args.grouped_metrics)
        fig_comparative_data_scaling(grouped_df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
