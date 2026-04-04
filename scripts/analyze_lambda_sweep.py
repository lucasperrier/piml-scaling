"""Aggregate and analyze lambda_phys sweep results.

Reads all metrics.json from the lambda sweep runs, compares against
matched plain-model results from the main runs, and produces a summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_sweep_metrics(sweep_root: Path) -> list[dict]:
    records = []
    for mpath in sorted(sweep_root.rglob("metrics.json")):
        with mpath.open() as f:
            m = json.load(f)
        # Extract lambda from directory path
        for part in mpath.parts:
            if part.startswith("lambda="):
                m["lambda_phys"] = float(part.split("=", 1)[1])
                break
        records.append(m)
    return records


def _load_plain_metrics(runs_root: Path) -> list[dict]:
    records = []
    for mpath in sorted((runs_root / "model=plain").rglob("metrics.json")):
        with mpath.open() as f:
            m = json.load(f)
        m["lambda_phys"] = -1.0  # sentinel for plain
        records.append(m)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=str, default="runs-lambda-sweep")
    parser.add_argument("--runs-root", type=str, default="runs", help="Main runs dir for plain-model comparison")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out) if args.out else sweep_root

    # Load sweep results
    sweep_records = _load_sweep_metrics(sweep_root)
    if not sweep_records:
        print(f"No metrics.json found under {sweep_root}")
        return

    # Load plain baselines
    plain_records = _load_plain_metrics(runs_root)

    df_sweep = pd.DataFrame(sweep_records)
    df_plain = pd.DataFrame(plain_records) if plain_records else pd.DataFrame()

    # Save raw aggregate
    agg_path = out_dir / "lambda_sweep_aggregate.csv"
    cols = [
        "lambda_phys", "model_name", "capacity_name", "parameter_count",
        "dataset_size", "data_seed", "train_seed", "status",
        "test_rel_l2", "val_rel_l2", "train_rel_l2",
        "test_mse", "val_mse", "train_mse",
        "best_epoch", "runtime_seconds",
    ]
    save_cols = [c for c in cols if c in df_sweep.columns]
    df_sweep[save_cols].to_csv(agg_path, index=False)
    print(f"Saved sweep aggregate: {agg_path} ({len(df_sweep)} runs)")

    # Grouped summary by (lambda, capacity, dataset_size)
    group_keys = ["lambda_phys", "capacity_name", "parameter_count", "dataset_size"]
    rows = []
    eligible = df_sweep[df_sweep["status"] == "success"]
    for keys, grp in eligible.groupby(group_keys, dropna=False):
        km = dict(zip(group_keys, keys))
        s = grp["test_rel_l2"]
        km["n_runs"] = len(grp)
        km["test_rel_l2_mean"] = float(s.mean())
        km["test_rel_l2_std"] = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        km["test_rel_l2_stderr"] = float(s.std(ddof=1) / len(s)**0.5) if len(s) > 1 else float("nan")
        rows.append(km)

    # Add plain baselines if available
    if not df_plain.empty:
        plain_eligible = df_plain[df_plain["status"] == "success"]
        for keys, grp in plain_eligible.groupby(["capacity_name", "parameter_count", "dataset_size"], dropna=False):
            cap, nparams, D = keys
            s = grp["test_rel_l2"]
            rows.append({
                "lambda_phys": -1.0,
                "capacity_name": cap,
                "parameter_count": nparams,
                "dataset_size": D,
                "n_runs": len(grp),
                "test_rel_l2_mean": float(s.mean()),
                "test_rel_l2_std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
                "test_rel_l2_stderr": float(s.std(ddof=1) / len(s)**0.5) if len(s) > 1 else float("nan"),
            })

    grouped = pd.DataFrame(rows).sort_values(["capacity_name", "dataset_size", "lambda_phys"]).reset_index(drop=True)
    grouped_path = out_dir / "lambda_sweep_grouped.csv"
    grouped.to_csv(grouped_path, index=False)
    print(f"Saved grouped summary: {grouped_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("LAMBDA SWEEP SUMMARY: test_rel_l2 (mean ± stderr)")
    print("=" * 80)

    for cap in sorted(grouped["capacity_name"].unique()):
        print(f"\n--- {cap} ---")
        cap_df = grouped[grouped["capacity_name"] == cap]
        for D in sorted(cap_df["dataset_size"].unique()):
            d_df = cap_df[cap_df["dataset_size"] == D].sort_values("lambda_phys")
            print(f"  D={D}:")
            for _, row in d_df.iterrows():
                lam = row["lambda_phys"]
                label = "plain" if lam < 0 else f"λ={lam}"
                mean = row["test_rel_l2_mean"]
                se = row.get("test_rel_l2_stderr", float("nan"))
                n = int(row["n_runs"])
                print(f"    {label:>12s}: {mean:.6f} ± {se:.6f}  (n={n})")

    # Lambda=0 vs plain comparison
    print("\n" + "=" * 80)
    print("LAMBDA=0 vs PLAIN EQUIVALENCE CHECK")
    print("=" * 80)
    lam0 = grouped[grouped["lambda_phys"] == 0.0]
    plain = grouped[grouped["lambda_phys"] < 0]
    if not lam0.empty and not plain.empty:
        merged = lam0.merge(
            plain[["capacity_name", "dataset_size", "test_rel_l2_mean"]],
            on=["capacity_name", "dataset_size"],
            suffixes=("_lam0", "_plain"),
            how="inner",
        )
        if not merged.empty:
            for _, row in merged.iterrows():
                ratio = row["test_rel_l2_mean_lam0"] / max(row["test_rel_l2_mean_plain"], 1e-15)
                status = "OK" if 0.9 < ratio < 1.1 else "MISMATCH"
                print(f"  {row['capacity_name']} D={int(row['dataset_size'])}: "
                      f"λ=0={row['test_rel_l2_mean_lam0']:.6f}  plain={row['test_rel_l2_mean_plain']:.6f}  "
                      f"ratio={ratio:.3f}  [{status}]")
        else:
            print("  No matching (capacity, D) found between λ=0 and plain.")
    else:
        print("  Missing λ=0 or plain data for comparison.")


if __name__ == "__main__":
    main()
