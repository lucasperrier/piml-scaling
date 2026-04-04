from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from scaling_piml.config_loader import load_experiment_config
from scaling_piml.data.dataset import FlowMapDataset
from scaling_piml.models.mlp import CAPACITY_GRID
from scaling_piml.train import train_one_run
from scaling_piml.utils.io import ensure_dir, save_json


def _parse_csv_ints(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return list(default)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv_strings(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _pilot_dataset_sizes(dataset_sizes: list[int]) -> list[int]:
    if len(dataset_sizes) <= 3:
        return list(dataset_sizes)
    picks = [dataset_sizes[0], dataset_sizes[len(dataset_sizes) // 2], dataset_sizes[-1]]
    seen: set[int] = set()
    ordered: list[int] = []
    for value in picks:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _run_dir(out_root: Path, model: str, capacity_name: str, dataset_size: int, data_seed: int, train_seed: int) -> Path:
    return ensure_dir(
        out_root
        / f"model={model}"
        / f"capacity={capacity_name}"
        / f"D={dataset_size}"
        / f"data_seed={data_seed}"
        / f"train_seed={train_seed}"
    )


def _failure_metrics(*, model: str, capacity_name: str, hidden_widths: list[int], dataset_size: int, data_seed: int, train_seed: int, data_root: Path, run_dir: Path, reason: str) -> dict[str, object]:
    config_path = run_dir / "config.yaml"
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"
    return {
        "model_name": model,
        "is_physics_informed": model == "piml",
        "capacity_name": capacity_name,
        "hidden_widths": hidden_widths,
        "parameter_count": -1,
        "dataset_size": dataset_size,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "status": "failed",
        "failure_reason": reason,
        "best_epoch": -1,
        "train_rel_l2": float("nan"),
        "val_rel_l2": float("nan"),
        "test_rel_l2": float("nan"),
        "train_mse": float("nan"),
        "val_mse": float("nan"),
        "test_mse": float("nan"),
        "runtime_seconds": float("nan"),
        "diverged": False,
        "nan_detected": False,
        "eligible_for_fit": False,
        "data_root": str(data_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "history_path": str(history_path.resolve()) if history_path.exists() else "",
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path.exists() else "",
        "metrics_path": str(metrics_path.resolve()),
        "device": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing data_seed=*/")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names from {plain,piml}")
    parser.add_argument("--capacities", type=str, default=None, help="Comma-separated capacity names")
    parser.add_argument("--dataset-sizes", type=str, default=None, help="Comma-separated dataset sizes")
    parser.add_argument("--data-seeds", type=str, default=None, help="Comma-separated data seeds")
    parser.add_argument("--train-seeds", type=str, default=None, help="Comma-separated train seeds")
    parser.add_argument("--pilot", action="store_true", help="Run the recommended pilot subset")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if metrics.json already exists")
    parser.add_argument("--lambda-phys", type=float, default=None, help="Override lambda_phys for PIML runs")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    out_root = Path(args.out) if args.out else Path(cfg.out_dir)
    data_dir = Path(args.data_dir)

    models = _parse_csv_strings(args.models, ["plain", "piml"])
    capacities = _parse_csv_strings(args.capacities, list(CAPACITY_GRID.keys()))
    dataset_sizes = _parse_csv_ints(args.dataset_sizes, cfg.data.dataset_sizes)
    data_seeds = _parse_csv_ints(args.data_seeds, cfg.data.data_seeds)
    train_seeds = _parse_csv_ints(args.train_seeds, cfg.train.train_seeds)

    if args.pilot:
        models = models[:2]
        capacities = capacities[:2]
        dataset_sizes = _pilot_dataset_sizes(dataset_sizes)
        data_seeds = data_seeds[:1]
        train_seeds = train_seeds[:2]

    total_runs = len(models) * len(capacities) * len(dataset_sizes) * len(data_seeds) * len(train_seeds)
    run_index = 0

    for model in models:
        if model not in {"plain", "piml"}:
            raise ValueError(f"Unknown model: {model}")
        for capacity_name in capacities:
            if capacity_name not in CAPACITY_GRID:
                raise ValueError(f"Unknown capacity: {capacity_name}")
            for dataset_size in dataset_sizes:
                for data_seed in data_seeds:
                    data_root = data_dir / f"data_seed={data_seed}"
                    if not data_root.exists():
                        raise FileNotFoundError(f"Missing dataset directory: {data_root}")
                    for train_seed in train_seeds:
                        run_index += 1
                        run_dir = _run_dir(out_root, model, capacity_name, dataset_size, data_seed, train_seed)
                        metrics_path = run_dir / "metrics.json"
                        if metrics_path.exists() and not args.overwrite:
                            print(f"[{run_index}/{total_runs}] skip {metrics_path}")
                            continue

                        run_cfg = deepcopy(cfg)
                        run_cfg.model.hidden_widths = CAPACITY_GRID[capacity_name]
                        if args.lambda_phys is not None:
                            run_cfg.train.lambda_phys = args.lambda_phys

                        train_ds = FlowMapDataset(data_root, "train", D=dataset_size, normalize=True)
                        val_ds = FlowMapDataset(data_root, "val", normalize=True)
                        test_ds = FlowMapDataset(data_root, "test", normalize=True)

                        try:
                            metrics = train_one_run(
                                cfg=run_cfg,
                                run_dir=run_dir,
                                model_name=model,
                                capacity_name=capacity_name,
                                is_physics_informed=model == "piml",
                                train_seed=train_seed,
                                data_root=data_root,
                                dataset_size=dataset_size,
                                train_dataset=train_ds,
                                val_dataset=val_ds,
                                test_dataset=test_ds,
                            )
                            print(
                                f"[{run_index}/{total_runs}] {model} {capacity_name} D={dataset_size} seed={data_seed}/{train_seed} -> {metrics['status']}"
                            )
                        except Exception as exc:
                            failure = _failure_metrics(
                                model=model,
                                capacity_name=capacity_name,
                                hidden_widths=CAPACITY_GRID[capacity_name],
                                dataset_size=dataset_size,
                                data_seed=data_seed,
                                train_seed=train_seed,
                                data_root=data_root,
                                run_dir=run_dir,
                                reason=str(exc),
                            )
                            save_json(metrics_path, failure)
                            print(
                                f"[{run_index}/{total_runs}] {model} {capacity_name} D={dataset_size} seed={data_seed}/{train_seed} -> failed: {exc}"
                            )


if __name__ == "__main__":
    main()