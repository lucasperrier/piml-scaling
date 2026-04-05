"""Run rescue variants for conservation prior bimodality diagnosis.

Usage:
    python scripts/run_rescue_sweep.py \\
        --data-dir data \\
        --out runs-rescue \\
        --capacities large,xlarge \\
        --dataset-sizes 256,512,1024 \\
        --lambda-phys 0.01,0.1,1.0

For each target configuration, runs 5 variants × 3 train seeds:
  1. Baseline conservation training
  2. Warm start from plain checkpoint
  3. Gradient clipping (max_norm=1.0)
  4. Lambda ramp-up over 50 epochs
  5. Two-stage (200 epochs data-only, then 200 epochs with conservation)
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


RESCUE_VARIANTS = {
    "baseline": {},
    "warm-start": {"warm_start": True},  # resolved per-config
    "grad-clip": {"grad_clip": 1.0},
    "lambda-ramp": {"lambda_schedule_epochs": 50},
    "two-stage": {"two_stage_epochs": 200},
}


def _parse_csv(raw: str | None, default: list) -> list:
    if raw is None:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _find_plain_checkpoint(
    out_root: Path, capacity_name: str, dataset_size: int, data_seed: int, train_seed: int
) -> str:
    """Look for a plain model checkpoint to use as warm start."""
    # Check common run directories
    for runs_dir in ["runs-dense", "runs-progress", "runs"]:
        ckpt = (
            Path(runs_dir)
            / f"model=plain"
            / f"capacity={capacity_name}"
            / f"D={dataset_size}"
            / f"data_seed={data_seed}"
            / f"train_seed={train_seed}"
            / "best.pt"
        )
        if ckpt.exists():
            return str(ckpt)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rescue variants for conservation prior")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="runs-rescue")
    parser.add_argument("--capacities", type=str, default="large,xlarge",
                        help="Comma-separated capacity names to target")
    parser.add_argument("--dataset-sizes", type=str, default="256,512,1024",
                        help="Comma-separated dataset sizes")
    parser.add_argument("--data-seeds", type=str, default=None)
    parser.add_argument("--train-seeds", type=str, default=None)
    parser.add_argument("--lambda-phys", type=str, default="0.01,0.1,1.0",
                        help="Comma-separated lambda_phys values")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    out_root = Path(args.out)
    data_dir = Path(args.data_dir)

    capacities = _parse_csv(args.capacities, ["large", "xlarge"])
    dataset_sizes = [int(x) for x in _parse_csv(args.dataset_sizes, [256, 512, 1024])]
    data_seeds = [int(x) for x in _parse_csv(args.data_seeds, cfg.data.data_seeds)]
    train_seeds = [int(x) for x in _parse_csv(args.train_seeds, cfg.train.train_seeds)]
    lambda_values = [float(x) for x in _parse_csv(args.lambda_phys, [0.01, 0.1, 1.0])]

    total = (
        len(capacities) * len(dataset_sizes) * len(lambda_values)
        * len(data_seeds) * len(train_seeds) * len(RESCUE_VARIANTS)
    )
    run_index = 0

    for cap_name in capacities:
        for ds in dataset_sizes:
            for lam in lambda_values:
                for data_seed in data_seeds:
                    data_root = data_dir / f"data_seed={data_seed}"
                    for train_seed in train_seeds:
                        for variant_name, variant_opts in RESCUE_VARIANTS.items():
                            run_index += 1
                            run_dir = ensure_dir(
                                out_root
                                / f"variant={variant_name}"
                                / f"lambda={lam}"
                                / f"capacity={cap_name}"
                                / f"D={ds}"
                                / f"data_seed={data_seed}"
                                / f"train_seed={train_seed}"
                            )
                            metrics_path = run_dir / "metrics.json"
                            if metrics_path.exists() and not args.overwrite:
                                print(f"[{run_index}/{total}] skip {run_dir.name}")
                                continue

                            run_cfg = deepcopy(cfg)
                            run_cfg.model.hidden_widths = CAPACITY_GRID[cap_name]
                            run_cfg.train.lambda_phys = lam

                            # Apply variant-specific options
                            if "grad_clip" in variant_opts:
                                run_cfg.train.grad_clip = variant_opts["grad_clip"]
                            if "lambda_schedule_epochs" in variant_opts:
                                run_cfg.train.lambda_schedule_epochs = variant_opts["lambda_schedule_epochs"]
                            if "two_stage_epochs" in variant_opts:
                                run_cfg.train.two_stage_epochs = variant_opts["two_stage_epochs"]
                            if variant_opts.get("warm_start"):
                                ckpt = _find_plain_checkpoint(
                                    out_root, cap_name, ds, data_seed, train_seed
                                )
                                run_cfg.train.warm_start = ckpt

                            train_ds = FlowMapDataset(data_root, "train", D=ds, normalize=True)
                            val_ds = FlowMapDataset(data_root, "val", normalize=True)
                            test_ds = FlowMapDataset(data_root, "test", normalize=True)

                            try:
                                metrics = train_one_run(
                                    cfg=run_cfg,
                                    run_dir=run_dir,
                                    model_name="piml-conservation",
                                    capacity_name=cap_name,
                                    physics_prior="conservation",
                                    train_seed=train_seed,
                                    data_root=data_root,
                                    dataset_size=ds,
                                    train_dataset=train_ds,
                                    val_dataset=val_ds,
                                    test_dataset=test_ds,
                                )
                                print(
                                    f"[{run_index}/{total}] {variant_name} λ={lam} {cap_name} "
                                    f"D={ds} seed={data_seed}/{train_seed} -> {metrics['status']}"
                                )
                            except Exception as exc:
                                print(
                                    f"[{run_index}/{total}] {variant_name} λ={lam} {cap_name} "
                                    f"D={ds} seed={data_seed}/{train_seed} -> FAILED: {exc}"
                                )

    print(f"\nRescue sweep complete. Results in {out_root}/")


if __name__ == "__main__":
    main()
