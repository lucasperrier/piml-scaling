"""Run scaling sweeps across multiple horizons T.

Usage:
    python scripts/run_horizon_sweep.py \\
        --horizons 0.5 1.0 2.0 \\
        --models plain piml piml-simpson \\
        --data-base-dir data-horizon \\
        --out-base-dir runs-horizon \\
        --generate-data

Generates data at each horizon (if --generate-data), then runs the full sweep
for each horizon × model combination.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sweeps across multiple horizons")
    parser.add_argument(
        "--horizons", type=float, nargs="+", default=[0.5, 1.0, 2.0],
        help="Horizon values T to sweep over",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--data-base-dir", type=str, default="data-horizon",
        help="Base directory for horizon data. Data is stored in {base}/T={horizon}/",
    )
    parser.add_argument(
        "--out-base-dir", type=str, default="runs-horizon",
        help="Base output directory. Runs stored in {base}/T={horizon}/",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["plain", "piml", "piml-simpson"],
        help="Models to include in the sweep",
    )
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Generate datasets at each horizon before sweeping",
    )
    parser.add_argument("--capacities", type=str, default=None)
    parser.add_argument("--dataset-sizes", type=str, default=None)
    parser.add_argument("--data-seeds", type=str, default=None)
    parser.add_argument("--train-seeds", type=str, default=None)
    parser.add_argument("--lambda-phys", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_base = Path(args.data_base_dir)
    out_base = Path(args.out_base_dir)

    for T in args.horizons:
        T_label = f"T={T}"
        data_dir = data_base / T_label
        out_dir = out_base / T_label

        print(f"\n{'='*60}")
        print(f"Horizon T={T}")
        print(f"{'='*60}")

        # Step 1: Generate data if requested
        if args.generate_data:
            data_dir.mkdir(parents=True, exist_ok=True)
            # Check if data already exists
            existing = list(data_dir.glob("data_seed=*"))
            if existing:
                print(f"  Data already exists at {data_dir} ({len(existing)} seeds), skipping generation")
            else:
                print(f"  Generating data at T={T} -> {data_dir}")
                cmd = [
                    sys.executable, "scripts/generate_datasets.py",
                    "--config", args.config,
                    "--out", str(data_dir),
                    "--horizon", str(T),
                ]
                subprocess.run(cmd, check=True)
                print(f"  Data generated at {data_dir}")

        # Step 2: Run sweep for each model
        for model in args.models:
            print(f"\n  Running sweep: model={model}, T={T}")
            cmd = [
                sys.executable, "scripts/run_sweep.py",
                "--config", args.config,
                "--data-dir", str(data_dir),
                "--out", str(out_dir),
                "--models", model,
                "--horizon", str(T),
            ]
            if args.capacities:
                cmd.extend(["--capacities", args.capacities])
            if args.dataset_sizes:
                cmd.extend(["--dataset-sizes", args.dataset_sizes])
            if args.data_seeds:
                cmd.extend(["--data-seeds", args.data_seeds])
            if args.train_seeds:
                cmd.extend(["--train-seeds", args.train_seeds])
            if args.lambda_phys is not None:
                cmd.extend(["--lambda-phys", str(args.lambda_phys)])
            if args.overwrite:
                cmd.append("--overwrite")
            subprocess.run(cmd, check=True)

    print(f"\nAll horizon sweeps complete. Results in {out_base}/")


if __name__ == "__main__":
    main()
