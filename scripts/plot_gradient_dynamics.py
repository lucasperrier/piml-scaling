"""Plot gradient dynamics decomposition from training history CSVs.

Usage:
    python scripts/plot_gradient_dynamics.py --run-dirs runs-grad-dynamics/model=plain/... \
        runs-grad-dynamics/model=piml/... runs-grad-dynamics/model=piml-simpson/... \
        --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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


def load_run(run_dir: Path) -> tuple[str, pd.DataFrame]:
    """Load training history and model name from a run directory."""
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.csv"

    with metrics_path.open() as f:
        metrics = json.load(f)

    model_name = metrics.get("model_name", "unknown")
    df = pd.read_csv(history_path)
    return model_name, df


def plot_grad_decomposition(run_dirs: list[Path], out_dir: Path) -> None:
    """Plot grad_norm_data and grad_norm_phys vs epoch for multiple runs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    ax = axes

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not (run_dir / "history.csv").exists():
            print(f"  Skipping {run_dir} (no history.csv)")
            continue

        model_name, df = load_run(run_dir)
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)

        if "grad_norm_data" in df.columns and "grad_norm_phys" in df.columns:
            ax.plot(df["epoch"], df["grad_norm_data"], "-",
                    color=color, alpha=0.8, label=f"{label} — data grad")
            ax.plot(df["epoch"], df["grad_norm_phys"], "--",
                    color=color, alpha=0.8, label=f"{label} — phys grad")
        else:
            ax.plot(df["epoch"], df["grad_norm"], "-",
                    color=color, alpha=0.8, label=f"{label} — total grad")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm")
    ax.set_yscale("log")
    ax.set_title("Gradient Dynamics: Data vs Physics Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_gradient_dynamics.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_gradient_dynamics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_gradient_dynamics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gradient dynamics decomposition")
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Paths to run directories containing history.csv with grad columns")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    plot_grad_decomposition([Path(d) for d in args.run_dirs], Path(args.out_dir))


if __name__ == "__main__":
    main()
