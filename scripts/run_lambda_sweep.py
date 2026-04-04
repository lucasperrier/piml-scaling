"""Run a lambda_phys sweep for PIML physics-loss diagnosis.

Runs PIML model across a grid of lambda_phys values on a representative
subset of capacities, dataset sizes, data seeds, and train seeds.
Also runs lambda_phys=0 to verify equivalence with the plain model.

Output directory structure:
  {out}/lambda={lambda_phys}/model=piml/capacity={cap}/D={D}/data_seed={ds}/train_seed={ts}/
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from scaling_piml.config_loader import load_experiment_config
from scaling_piml.data.dataset import FlowMapDataset
from scaling_piml.models.mlp import CAPACITY_GRID
from scaling_piml.train import train_one_run
from scaling_piml.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda sweep for PIML diagnosis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="runs-lambda-sweep")
    parser.add_argument(
        "--lambdas", type=str, default="0,1e-4,1e-3,1e-2,1e-1,1",
        help="Comma-separated lambda_phys values",
    )
    parser.add_argument(
        "--capacities", type=str, default="small,large",
        help="Comma-separated capacity names",
    )
    parser.add_argument(
        "--dataset-sizes", type=str, default="128,1024,4096",
        help="Comma-separated dataset sizes",
    )
    parser.add_argument(
        "--data-seeds", type=str, default="11,22,33",
        help="Comma-separated data seeds",
    )
    parser.add_argument(
        "--train-seeds", type=str, default="101,202,303",
        help="Comma-separated train seeds",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    data_dir = Path(args.data_dir)
    out_root = Path(args.out)

    lambdas = [float(x.strip()) for x in args.lambdas.split(",")]
    capacities = [x.strip() for x in args.capacities.split(",")]
    dataset_sizes = [int(x.strip()) for x in args.dataset_sizes.split(",")]
    data_seeds = [int(x.strip()) for x in args.data_seeds.split(",")]
    train_seeds = [int(x.strip()) for x in args.train_seeds.split(",")]

    total = len(lambdas) * len(capacities) * len(dataset_sizes) * len(data_seeds) * len(train_seeds)
    idx = 0

    for lam in lambdas:
        for cap in capacities:
            for D in dataset_sizes:
                for dseed in data_seeds:
                    data_root = data_dir / f"data_seed={dseed}"
                    if not data_root.exists():
                        print(f"SKIP missing {data_root}")
                        idx += len(train_seeds)
                        continue
                    for tseed in train_seeds:
                        idx += 1
                        run_dir = ensure_dir(
                            out_root
                            / f"lambda={lam}"
                            / f"model=piml"
                            / f"capacity={cap}"
                            / f"D={D}"
                            / f"data_seed={dseed}"
                            / f"train_seed={tseed}"
                        )
                        metrics_path = run_dir / "metrics.json"
                        if metrics_path.exists() and not args.overwrite:
                            print(f"[{idx}/{total}] skip {run_dir.name}")
                            continue

                        run_cfg = deepcopy(cfg)
                        run_cfg.model.hidden_widths = CAPACITY_GRID[cap]
                        run_cfg.train.lambda_phys = lam

                        train_ds = FlowMapDataset(data_root, "train", D=D, normalize=True)
                        val_ds = FlowMapDataset(data_root, "val", normalize=True)
                        test_ds = FlowMapDataset(data_root, "test", normalize=True)

                        try:
                            metrics = train_one_run(
                                cfg=run_cfg,
                                run_dir=run_dir,
                                model_name="piml",
                                capacity_name=cap,
                                is_physics_informed=(lam > 0),
                                train_seed=tseed,
                                data_root=data_root,
                                dataset_size=D,
                                train_dataset=train_ds,
                                val_dataset=val_ds,
                                test_dataset=test_ds,
                            )
                            # Add lambda to metrics for downstream analysis
                            metrics["lambda_phys"] = lam
                            save_json(metrics_path, metrics)
                            print(
                                f"[{idx}/{total}] λ={lam} {cap} D={D} seed={dseed}/{tseed} "
                                f"-> {metrics['status']} test_rel_l2={metrics['test_rel_l2']:.6f}"
                            )
                        except Exception as exc:
                            print(f"[{idx}/{total}] λ={lam} {cap} D={D} seed={dseed}/{tseed} -> FAILED: {exc}")


if __name__ == "__main__":
    main()
