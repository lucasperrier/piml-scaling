from __future__ import annotations

import argparse
from pathlib import Path

from scaling_piml.config_loader import load_experiment_config
from scaling_piml.data.dataset import FlowMapDataset
from scaling_piml.train import train_one_run
from scaling_piml.utils.io import ensure_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data-root", type=str, required=True, help="Path like data/data_seed=11")
    p.add_argument("--D", type=int, required=True)
    p.add_argument("--train-seed", type=int, required=True)

    p.add_argument("--model", type=str, choices=["plain", "piml"], default="plain")
    p.add_argument(
        "--capacity",
        type=str,
        default=None,
        help="Optional capacity name (e.g. small) to override hidden widths",
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--lambda-phys", type=float, default=None, help="Override lambda_phys")

    args = p.parse_args()

    cfg = load_experiment_config(args.config)

    if args.lambda_phys is not None:
        cfg.train.lambda_phys = args.lambda_phys

    if args.capacity is not None:
        from scaling_piml.models.mlp import CAPACITY_GRID

        cfg.model.hidden_widths = CAPACITY_GRID[args.capacity]

    is_piml = args.model == "piml"
    model_name = args.model
    capacity_name = args.capacity or "custom"

    train_ds = FlowMapDataset(args.data_root, "train", D=args.D, normalize=True)
    val_ds = FlowMapDataset(args.data_root, "val", normalize=True)
    test_ds = FlowMapDataset(args.data_root, "test", normalize=True)

    out_root = Path(args.out) if args.out else Path(cfg.out_dir)
    run_dir = ensure_dir(
        out_root
        / f"model={args.model}"
        / f"capacity={capacity_name if args.capacity else ','.join(map(str, cfg.model.hidden_widths))}"
        / f"D={args.D}"
        / f"data_seed={Path(args.data_root).name.split('=')[-1]}"
        / f"train_seed={args.train_seed}"
    )

    metrics = train_one_run(
        cfg=cfg,
        run_dir=run_dir,
        model_name=model_name,
        capacity_name=capacity_name,
        is_physics_informed=is_piml,
        train_seed=args.train_seed,
        data_root=args.data_root,
        dataset_size=args.D,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )

    print(metrics)


if __name__ == "__main__":
    main()
