"""Merge grouped_metrics.csv files from multiple run directories.

Usage:
    python scripts/merge_grouped_metrics.py \
        --inputs runs-progress/grouped_metrics.csv runs-simpson/grouped_metrics.csv \
        --out-dir runs-combined
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge grouped_metrics from multiple run dirs")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, default="runs-combined")
    args = parser.parse_args()

    frames = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} does not exist, skipping")
            continue
        df = pd.read_csv(path)
        # Ensure physics_prior column exists
        if "physics_prior" not in df.columns:
            df["physics_prior"] = df["model_name"].map(
                lambda m: "none" if m == "plain" else m.replace("piml-", "") if "-" in m else "midpoint"
            )
        print(f"  Loaded {path}: {len(df)} rows, models={sorted(df['model_name'].unique())}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid input files found")

    combined = pd.concat(frames, ignore_index=True)

    # Drop duplicates (same model/capacity/dataset_size)
    dedup_keys = ["model_name", "capacity_name", "dataset_size"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    if len(combined) < before:
        print(f"  Dropped {before - len(combined)} duplicate rows")

    combined = combined.sort_values(
        ["model_name", "capacity_name", "dataset_size"]
    ).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grouped_metrics.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows)")
    print(f"  Models: {sorted(combined['model_name'].unique())}")


if __name__ == "__main__":
    main()
