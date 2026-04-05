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

    p.add_argument("--model", type=str, choices=["plain", "piml", "piml-conservation", "piml-simpson", "piml-simpson-true"], default="plain")
    p.add_argument(
        "--capacity",
        type=str,
        default=None,
        help="Optional capacity name (e.g. small) to override hidden widths",
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--lambda-phys", type=float, default=None, help="Override lambda_phys")
    p.add_argument("--horizon", type=float, default=None, help="Override data.T (horizon)")
    p.add_argument("--warm-start", type=str, default=None, help="Path to plain-model checkpoint for warm start")
    p.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm for clipping")
    p.add_argument("--lambda-schedule", type=int, default=None, help="Linear ramp-up epochs for lambda_phys")
    p.add_argument("--two-stage", type=int, default=None, help="Data-only epochs before adding physics loss")
    p.add_argument("--save-preds", action="store_true", help="Save test predictions to run directory")
    p.add_argument("--system", type=str, default=None, choices=["lotka-volterra", "duffing"],
                    help="Override system name in config")
    p.add_argument("--obs-noise", type=float, default=0.0,
                    help="Gaussian noise std added to training targets (fraction of per-component std)")
    p.add_argument("--prior-params", type=str, default=None,
                    help="Override system params in physics loss only (comma-sep: alpha,beta,delta,gamma)")
    p.add_argument("--log-grad-decomposition", action="store_true",
                    help="Log grad_norm_data and grad_norm_phys separately per epoch")

    args = p.parse_args()

    cfg = load_experiment_config(args.config)

    if args.lambda_phys is not None:
        cfg.train.lambda_phys = args.lambda_phys

    if args.horizon is not None:
        cfg.data.T = args.horizon

    if args.warm_start is not None:
        cfg.train.warm_start = args.warm_start
    if args.grad_clip is not None:
        cfg.train.grad_clip = args.grad_clip
    if args.lambda_schedule is not None:
        cfg.train.lambda_schedule_epochs = args.lambda_schedule
    if args.two_stage is not None:
        cfg.train.two_stage_epochs = args.two_stage
    if args.system is not None:
        cfg.system.name = args.system
    if args.log_grad_decomposition:
        cfg.train.log_grad_decomposition = True

    if args.capacity is not None:
        from scaling_piml.models.mlp import CAPACITY_GRID

        cfg.model.hidden_widths = CAPACITY_GRID[args.capacity]

    _PRIOR_MAP = {"plain": "none", "piml": "midpoint", "piml-conservation": "conservation", "piml-simpson": "simpson", "piml-simpson-true": "simpson-true"}
    physics_prior = _PRIOR_MAP[args.model]
    model_name = args.model
    capacity_name = args.capacity or "custom"

    train_ds = FlowMapDataset(args.data_root, "train", D=args.D, normalize=True,
                               obs_noise=args.obs_noise, noise_seed=args.train_seed)
    val_ds = FlowMapDataset(args.data_root, "val", normalize=True)
    test_ds = FlowMapDataset(args.data_root, "test", normalize=True)

    # Prior mismatch: override system params used in physics loss
    if args.prior_params is not None:
        parts = [float(v) for v in args.prior_params.split(",")]
        cfg.system.alpha = parts[0]
        cfg.system.beta = parts[1]
        if len(parts) > 2:
            cfg.system.delta = parts[2]
        if len(parts) > 3:
            cfg.system.gamma = parts[3]

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
        physics_prior=physics_prior,
        train_seed=args.train_seed,
        data_root=args.data_root,
        dataset_size=args.D,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        save_predictions=args.save_preds,
    )

    print(metrics)


if __name__ == "__main__":
    main()
