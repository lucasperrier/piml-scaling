"""Generate all paper figures from aggregated results and scaling fits.

Usage:
    python scripts/generate_figures.py --grouped-metrics runs/grouped_metrics.csv \
        --scaling-fits runs/scaling_fits.json --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "plain": "#1f77b4",
    "piml": "#d62728",
    "piml-simpson": "#2ca02c",
    "piml-conservation": "#ff7f0e",
}
MODEL_LABELS = {
    "plain": "Plain MLP",
    "piml": "PIML (midpoint)",
    "piml-simpson": "PIML (composite)",
    "piml-conservation": "PIML (conservation)",
}
MARKERS = {"plain": "o", "piml": "s", "piml-simpson": "D", "piml-conservation": "^"}


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


# ---------------------------------------------------------------------------
# Figure 1: Task setup schematic
# ---------------------------------------------------------------------------

def fig_task_schematic(out_dir: Path) -> None:
    """Simple schematic of the Lotka–Volterra flow-map prediction task."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: phase portrait sketch
    ax = axes[0]
    t = np.linspace(0, 8, 500)
    # Approximate Lotka-Volterra orbit for illustration
    x = 1.5 + 0.8 * np.cos(t)
    y = 1.5 + 0.8 * np.sin(t * 1.1)
    ax.plot(x, y, "b-", lw=1.5, alpha=0.7)
    ax.plot(x[0], y[0], "go", ms=8, label="$u_0$", zorder=5)
    # mark T endpoint
    T_idx = 80
    ax.plot(x[T_idx], y[T_idx], "r^", ms=8, label="$u(T)$", zorder=5)
    ax.annotate("", xy=(x[T_idx], y[T_idx]), xytext=(x[0], y[0]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.set_xlabel("Prey $x$")
    ax.set_ylabel("Predator $y$")
    ax.set_title("Lotka–Volterra Phase Space")
    ax.legend()

    # Right: flow map diagram
    ax = axes[1]
    ax.text(0.1, 0.5, "$u_0$", fontsize=18, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.annotate("", xy=(0.55, 0.5), xytext=(0.25, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.4, 0.6, "$f_\\theta$", fontsize=14, ha="center")
    ax.text(0.7, 0.5, "$\\hat{u}(T)$", fontsize=18, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.text(0.4, 0.25, "$\\mathcal{L}_{\\mathrm{data}} + \\lambda\\,\\mathcal{L}_{\\mathrm{phys}}$",
            fontsize=12, ha="center")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Flow-Map Prediction")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_task_schematic.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_task_schematic.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_task_schematic")


# ---------------------------------------------------------------------------
# Figure 2: Capacity scaling – error vs N at fixed D
# ---------------------------------------------------------------------------

def fig_capacity_scaling(grouped_df: pd.DataFrame, scaling_fits: dict, out_dir: Path) -> None:
    df = grouped_df.copy()
    dataset_sizes = sorted(df["dataset_size"].unique())
    # Pick a few representative D values (small, medium, large)
    if len(dataset_sizes) > 4:
        picks = [dataset_sizes[0], dataset_sizes[len(dataset_sizes) // 2], dataset_sizes[-1]]
    else:
        picks = dataset_sizes

    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4), squeeze=False)

    all_models = sorted(df["model_name"].unique())
    for ax, D in zip(axes[0], picks):
        for model in all_models:
            sub = df[(df["model_name"] == model) & (df["dataset_size"] == D)].sort_values("parameter_count")
            if sub.empty:
                continue
            N = sub["parameter_count"].values
            E = sub["test_rel_l2_mean"].values
            E_err = sub["test_rel_l2_stderr"].values if "test_rel_l2_stderr" in sub.columns else np.zeros_like(E)
            ax.errorbar(N, E, yerr=E_err, label=MODEL_LABELS.get(model, model),
                        color=MODEL_COLORS.get(model, "gray"), marker=MARKERS.get(model, "x"), capsize=3)

            # overlay fit curve if available
            for fit_rec in scaling_fits.get("capacity_fits", []):
                if fit_rec["model_name"] == model and fit_rec["dataset_size"] == D and "alpha" in fit_rec:
                    N_fit = np.linspace(N.min() * 0.8, N.max() * 1.2, 100)
                    E_fit = fit_rec["E_inf"] + fit_rec["a"] * np.power(N_fit, -fit_rec["alpha"])
                    ax.plot(N_fit, E_fit, "--", color=MODEL_COLORS.get(model, "gray"), alpha=0.5,
                            label=f"$\\alpha$={fit_rec['alpha']:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameter count $N$")
        ax.set_ylabel("Test relative L2")
        ax.set_title(f"$D = {D}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Capacity Scaling: Error vs Model Size", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_capacity_scaling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_capacity_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_capacity_scaling")


# ---------------------------------------------------------------------------
# Figure 3: Data scaling – error vs D at fixed N
# ---------------------------------------------------------------------------

def fig_data_scaling(grouped_df: pd.DataFrame, scaling_fits: dict, out_dir: Path) -> None:
    df = grouped_df.copy()
    capacities = sorted(df["capacity_name"].unique())

    # Pick a few representative capacities
    if len(capacities) > 4:
        picks = [capacities[0], capacities[len(capacities) // 2], capacities[-1]]
    else:
        picks = capacities

    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4), squeeze=False)

    all_models = sorted(df["model_name"].unique())
    for ax, cap in zip(axes[0], picks):
        for model in all_models:
            sub = df[(df["model_name"] == model) & (df["capacity_name"] == cap)].sort_values("dataset_size")
            if sub.empty:
                continue
            D = sub["dataset_size"].values
            E = sub["test_rel_l2_mean"].values
            E_err = sub["test_rel_l2_stderr"].values if "test_rel_l2_stderr" in sub.columns else np.zeros_like(E)
            ax.errorbar(D, E, yerr=E_err, label=MODEL_LABELS.get(model, model),
                        color=MODEL_COLORS.get(model, "gray"), marker=MARKERS.get(model, "x"), capsize=3)

            # overlay fit curve
            for fit_rec in scaling_fits.get("data_fits", []):
                if fit_rec["model_name"] == model and fit_rec["capacity_name"] == cap and "beta" in fit_rec:
                    D_fit = np.linspace(D.min() * 0.8, D.max() * 1.2, 100)
                    E_fit = fit_rec["E_inf"] + fit_rec["b"] * np.power(D_fit, -fit_rec["beta"])
                    ax.plot(D_fit, E_fit, "--", color=MODEL_COLORS.get(model, "gray"), alpha=0.5,
                            label=f"$\\beta$={fit_rec['beta']:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dataset size $D$")
        ax.set_ylabel("Test relative L2")
        ax.set_title(f"Capacity: {cap}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Data Scaling: Error vs Dataset Size", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_data_scaling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_data_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_data_scaling")


# ---------------------------------------------------------------------------
# Figure 4: 2D error heatmap / contour
# ---------------------------------------------------------------------------

def fig_error_heatmap(grouped_df: pd.DataFrame, out_dir: Path) -> None:
    models = sorted(grouped_df["model_name"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = grouped_df[grouped_df["model_name"] == model].copy()
        pivot = sub.pivot_table(
            index="parameter_count", columns="dataset_size",
            values="test_rel_l2_mean", aggfunc="mean"
        )
        if pivot.empty:
            continue
        N_vals = pivot.index.values.astype(float)
        D_vals = pivot.columns.values.astype(float)
        Z = pivot.values

        im = ax.pcolormesh(
            np.log10(D_vals), np.log10(N_vals), Z,
            shading="auto", cmap="viridis"
        )
        fig.colorbar(im, ax=ax, label="Test rel. L2")
        ax.set_xlabel("$\\log_{10}(D)$")
        ax.set_ylabel("$\\log_{10}(N)$")
        ax.set_title(MODEL_LABELS.get(model, model))

    fig.suptitle("Error Surface: $E(N, D)$", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_error_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_error_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_error_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: Exponent comparison with CI
# ---------------------------------------------------------------------------

def fig_exponent_comparison(scaling_fits: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Alpha (capacity exponent) from full surface fits
    ax = axes[0]
    models = []
    alphas = []
    alpha_ci = []
    for fit in scaling_fits.get("full_fits", []):
        boot = fit.get("bootstrap", {})
        if boot.get("n_boot_success", 0) < 10:
            continue
        models.append(fit["model_name"])
        alphas.append(boot["alpha_mean"])
        alpha_ci.append([boot["alpha_mean"] - boot["alpha_ci_lo"],
                        boot["alpha_ci_hi"] - boot["alpha_mean"]])

    if models:
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        x_pos = range(len(models))
        for i, (m, a, ci) in enumerate(zip(models, alphas, alpha_ci)):
            ax.errorbar(i, a, yerr=[[ci[0]], [ci[1]]], fmt="o", color=colors[i],
                        capsize=8, ms=10, label=MODEL_LABELS.get(m, m))
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_ylabel("Capacity exponent $\\alpha$")
        ax.set_title("Capacity Scaling Exponent")
        ax.grid(True, alpha=0.3, axis="y")

    # Beta (data exponent) from full surface fits
    ax = axes[1]
    models = []
    betas = []
    beta_ci = []
    for fit in scaling_fits.get("full_fits", []):
        boot = fit.get("bootstrap", {})
        if boot.get("n_boot_success", 0) < 10:
            continue
        models.append(fit["model_name"])
        betas.append(boot["beta_mean"])
        beta_ci.append([boot["beta_mean"] - boot["beta_ci_lo"],
                       boot["beta_ci_hi"] - boot["beta_mean"]])

    if models:
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        x_pos = range(len(models))
        for i, (m, b, ci) in enumerate(zip(models, betas, beta_ci)):
            ax.errorbar(i, b, yerr=[[ci[0]], [ci[1]]], fmt="o", color=colors[i],
                        capsize=8, ms=10, label=MODEL_LABELS.get(m, m))
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_ylabel("Data exponent $\\beta$")
        ax.set_title("Data Scaling Exponent")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exponent Comparison with 95% CI", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exponent_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_exponent_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_exponent_comparison")


# ---------------------------------------------------------------------------
# Figure 6: Stability summary
# ---------------------------------------------------------------------------

def fig_stability_summary(grouped_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Divergence rate by model
    ax = axes[0]
    all_models = sorted(grouped_df["model_name"].unique())
    for model in all_models:
        sub = grouped_df[grouped_df["model_name"] == model].sort_values("dataset_size")
        if sub.empty:
            continue
        # Average divergence rate across capacities for each D
        agg = sub.groupby("dataset_size")["divergence_rate"].mean().reset_index()
        ax.plot(agg["dataset_size"], agg["divergence_rate"],
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "gray"), marker=MARKERS.get(model, "x"))
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size $D$")
    ax.set_ylabel("Divergence rate")
    ax.set_title("Divergence Rate vs Dataset Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    # Error spread (std) by model and D
    ax = axes[1]
    for model in all_models:
        sub = grouped_df[grouped_df["model_name"] == model].sort_values("dataset_size")
        if sub.empty:
            continue
        if "test_rel_l2_std" in sub.columns:
            agg = sub.groupby("dataset_size")["test_rel_l2_std"].mean().reset_index()
            ax.plot(agg["dataset_size"], agg["test_rel_l2_std"],
                    label=MODEL_LABELS.get(model, model),
                    color=MODEL_COLORS.get(model, "gray"), marker=MARKERS.get(model, "x"))
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size $D$")
    ax.set_ylabel("Std of test rel. L2 (across seeds)")
    ax.set_title("Seed Variance vs Dataset Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Stability Summary", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stability_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_stability_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_stability_summary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--grouped-metrics", type=str, required=True)
    parser.add_argument("--scaling-fits", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    _setup_style()

    grouped_df = pd.read_csv(args.grouped_metrics)
    with open(args.scaling_fits) as f:
        scaling_fits = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig_task_schematic(out_dir)
    fig_capacity_scaling(grouped_df, scaling_fits, out_dir)
    fig_data_scaling(grouped_df, scaling_fits, out_dir)
    fig_error_heatmap(grouped_df, out_dir)
    fig_exponent_comparison(scaling_fits, out_dir)
    fig_stability_summary(grouped_df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
