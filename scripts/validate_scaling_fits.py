"""Validate scaling-law ansatzes via held-out prediction and model comparison.

Usage:
    python scripts/validate_scaling_fits.py \\
        --grouped-metrics runs-progress/grouped_metrics.csv \\
        --out-dir runs-progress/ansatz_comparison \\
        --n-boot 1000

Runs all five ansatzes (A-E) on each model, reports:
  1. AIC/BIC ranking
  2. Leave-one-column-out (hold out one D) prediction error
  3. Leave-one-row-out (hold out one N) prediction error
  4. Leave-one-corner-out (extrapolation) prediction error
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scaling_piml.analysis.ansatz_comparison import (
    ALL_ANSATZES,
    get_ansatz,
    leave_one_column_out,
    leave_one_corner_out,
    leave_one_row_out,
    run_ansatz_comparison,
)


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _run_cv(
    df: pd.DataFrame,
    metric_col: str,
    max_divergence_rate: float,
) -> dict:
    """Run cross-validation for all models and ansatzes."""
    df = df.copy()
    df = df[df["divergence_rate"] <= max_divergence_rate]
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    cv_results: dict = {}
    for model_name, model_df in df.groupby("model_name"):
        N = model_df["parameter_count"].values.astype(float)
        D = model_df["dataset_size"].values.astype(float)
        E = model_df[metric_col].values.astype(float)

        model_cv: dict = {}
        for aname, ansatz in ALL_ANSATZES.items():
            col_out = leave_one_column_out(N, D, E, ansatz)
            row_out = leave_one_row_out(N, D, E, ansatz)
            corner_out = leave_one_corner_out(N, D, E, ansatz)

            # Summarize across folds
            col_rmses = [f["rmse"] for f in col_out["folds"]]
            row_rmses = [f["rmse"] for f in row_out["folds"]]
            corner_rmses = [f["rmse"] for f in corner_out["folds"]]

            model_cv[aname] = {
                "ansatz": ansatz.name,
                "column_out": {
                    "n_folds": len(col_rmses),
                    "mean_rmse": float(np.mean(col_rmses)) if col_rmses else None,
                    "folds": col_out["folds"],
                },
                "row_out": {
                    "n_folds": len(row_rmses),
                    "mean_rmse": float(np.mean(row_rmses)) if row_rmses else None,
                    "folds": row_out["folds"],
                },
                "corner_out": {
                    "n_folds": len(corner_rmses),
                    "mean_rmse": float(np.mean(corner_rmses)) if corner_rmses else None,
                    "folds": corner_out["folds"],
                },
            }
        cv_results[str(model_name)] = model_cv
    return cv_results


def _print_summary(comparison: dict, cv: dict) -> None:
    """Print human-readable summary to stdout."""
    for model, data in comparison.items():
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # AIC/BIC ranking
        ranking = data.get("ranking", [])
        if ranking:
            print("\n  AIC/BIC Ranking (lower is better):")
            print(f"  {'Ansatz':<25} {'BIC':>10} {'AIC':>10} {'k':>4}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*4}")
            for r in ranking:
                print(f"  {r['ansatz']:<25} {r['bic']:>10.2f} {r['aic']:>10.2f} {r['n_params']:>4}")

        # Fit parameters for ansatz A
        for fit in data.get("fits", []):
            if fit.get("ansatz") == "A_additive_floor" and fit.get("converged"):
                params = fit["params"]
                print(f"\n  Ansatz A params: E∞={params['E_inf']:.4f}, "
                      f"a={params['a']:.4f}, α={params['alpha']:.3f}, "
                      f"b={params['b']:.4f}, β={params['beta']:.3f}")

        # CV summary
        model_cv = cv.get(model, {})
        if model_cv:
            print("\n  Cross-validation mean RMSE:")
            print(f"  {'Ansatz':<25} {'Col-out':>10} {'Row-out':>10} {'Corner':>10}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
            for aname in ALL_ANSATZES:
                acv = model_cv.get(aname, {})
                col_rmse = acv.get("column_out", {}).get("mean_rmse")
                row_rmse = acv.get("row_out", {}).get("mean_rmse")
                corner_rmse = acv.get("corner_out", {}).get("mean_rmse")
                aobj = ALL_ANSATZES[aname]
                col_s = f"{col_rmse:>10.5f}" if col_rmse is not None else f"{'N/A':>10}"
                row_s = f"{row_rmse:>10.5f}" if row_rmse is not None else f"{'N/A':>10}"
                cor_s = f"{corner_rmse:>10.5f}" if corner_rmse is not None else f"{'N/A':>10}"
                print(f"  {aobj.name:<25} {col_s} {row_s} {cor_s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scaling-law ansatzes")
    parser.add_argument("--grouped-metrics", type=str, required=True,
                        help="Path to grouped_metrics.csv")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--metric", type=str, default="test_rel_l2_mean")
    parser.add_argument("--max-divergence-rate", type=float, default=0.30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=42)
    parser.add_argument("--ansatzes", type=str, nargs="*", default=None,
                        help="Ansatzes to compare (A B C D E). Default: all")
    args = parser.parse_args()

    grouped_path = Path(args.grouped_metrics)
    out_dir = Path(args.out_dir) if args.out_dir else grouped_path.parent / "ansatz_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_df = pd.read_csv(grouped_path)
    print(f"Loaded {len(grouped_df)} rows from {grouped_path}")

    # 1. Run ansatz comparison (fits + bootstrap)
    print("\nFitting ansatzes with bootstrap...")
    comparison = run_ansatz_comparison(
        grouped_df,
        metric_col=args.metric,
        max_divergence_rate=args.max_divergence_rate,
        ansatz_names=args.ansatzes,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
    )

    comp_path = out_dir / "ansatz_fits.json"
    with comp_path.open("w") as f:
        json.dump(comparison, f, indent=2, cls=_Encoder)
    print(f"Wrote {comp_path}")

    # 2. Run cross-validation
    print("\nRunning cross-validation...")
    cv_results = _run_cv(
        grouped_df,
        metric_col=args.metric,
        max_divergence_rate=args.max_divergence_rate,
    )

    cv_path = out_dir / "ansatz_cv.json"
    with cv_path.open("w") as f:
        json.dump(cv_results, f, indent=2, cls=_Encoder)
    print(f"Wrote {cv_path}")

    # 3. Print summary
    _print_summary(comparison, cv_results)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
