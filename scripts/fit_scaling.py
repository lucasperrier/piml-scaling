"""Fit scaling curves to aggregated experiment results.

Usage:
    python scripts/fit_scaling.py --grouped-metrics runs/grouped_metrics.csv --out-dir runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scaling_piml.analysis.scaling import run_scaling_analysis


class _Encoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np  # noqa: E402
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit scaling curves to grouped metrics")
    parser.add_argument("--grouped-metrics", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--metric", type=str, default="test_rel_l2_mean")
    parser.add_argument("--max-divergence-rate", type=float, default=0.30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=42)
    args = parser.parse_args()

    grouped_path = Path(args.grouped_metrics)
    out_dir = Path(args.out_dir) if args.out_dir else grouped_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_df = pd.read_csv(grouped_path)

    results = run_scaling_analysis(
        grouped_df,
        metric_col=args.metric,
        max_divergence_rate=args.max_divergence_rate,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
    )

    out_path = out_dir / "scaling_fits.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, cls=_Encoder)
    print(f"Wrote {out_path}")

    # Print summary
    for fit in results["full_fits"]:
        model = fit["model_name"]
        boot = fit.get("bootstrap", {})
        alpha_str = f"{boot.get('alpha_mean', float('nan')):.3f} [{boot.get('alpha_ci_lo', float('nan')):.3f}, {boot.get('alpha_ci_hi', float('nan')):.3f}]"
        beta_str = f"{boot.get('beta_mean', float('nan')):.3f} [{boot.get('beta_ci_lo', float('nan')):.3f}, {boot.get('beta_ci_hi', float('nan')):.3f}]"
        print(f"  {model}: alpha={alpha_str}  beta={beta_str}  n_boot_success={boot.get('n_boot_success', 0)}")


if __name__ == "__main__":
    main()
