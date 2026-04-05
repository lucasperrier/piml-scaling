from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DataConfig, ODESolverConfig, SystemConfig
from ..systems.lotka_volterra import lotka_volterra_rhs
from ..systems.duffing import duffing_rhs
from ..systems.ode import solve_flow_map
from ..utils.io import ensure_dir, save_json
from ..utils.seed import seed_everything


@dataclass
class NormalizationStats:
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]


def sample_initial_conditions(cfg: DataConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    x0 = rng.uniform(cfg.x0_low, cfg.x0_high, size=(n, 1))
    y0 = rng.uniform(cfg.y0_low, cfg.y0_high, size=(n, 1))
    return np.concatenate([x0, y0], axis=1).astype(np.float64)


def generate_dataset_for_seed(
    *,
    data_seed: int,
    out_dir: str | Path,
    system: SystemConfig,
    solver: ODESolverConfig,
    data: DataConfig,
    system_name: str = "lotka-volterra",
) -> Path:
    """Generate and freeze train/val/test split + nested train subsets + normalization."""

    seed_everything(data_seed)
    rng = np.random.default_rng(data_seed)

    root = ensure_dir(Path(out_dir) / f"data_seed={data_seed}")

    n_total = data.train_pool + data.val_size + data.test_size
    u0_all = sample_initial_conditions(data, n_total, rng=rng)

    if system_name == "duffing":
        def rhs(t: float, u: np.ndarray) -> np.ndarray:
            return duffing_rhs(
                t, u,
                alpha=system.alpha,
                beta=system.beta,
            )
    else:
        def rhs(t: float, u: np.ndarray) -> np.ndarray:
            return lotka_volterra_rhs(
                t,
                u,
                alpha=system.alpha,
                beta=system.beta,
                delta=system.delta,
                gamma=system.gamma,
            )

    uT_all = np.zeros_like(u0_all)
    for i in range(n_total):
        uT_all[i] = solve_flow_map(
            rhs,
            u0_all[i],
            T=data.T,
            method=solver.method,
            rtol=solver.rtol,
            atol=solver.atol,
        )

    idx = np.arange(n_total)
    rng.shuffle(idx)

    train_idx = idx[: data.train_pool]
    val_idx = idx[data.train_pool : data.train_pool + data.val_size]
    test_idx = idx[data.train_pool + data.val_size :]

    u0_train = u0_all[train_idx]
    uT_train = uT_all[train_idx]

    # Normalization from full train pool only
    x_mean = u0_train.mean(axis=0)
    x_std = u0_train.std(axis=0) + 1e-12
    y_mean = uT_train.mean(axis=0)
    y_std = uT_train.std(axis=0) + 1e-12

    np.save(root / "u0_all.npy", u0_all)
    np.save(root / "uT_all.npy", uT_all)
    np.save(root / "train_idx.npy", train_idx)
    np.save(root / "val_idx.npy", val_idx)
    np.save(root / "test_idx.npy", test_idx)

    # Nested subsets as first D of train_idx
    subsets: dict[str, list[int]] = {str(D): train_idx[:D].tolist() for D in data.dataset_sizes}
    save_json(root / "train_subsets.json", subsets)

    stats = NormalizationStats(
        x_mean=x_mean.tolist(),
        x_std=x_std.tolist(),
        y_mean=y_mean.tolist(),
        y_std=y_std.tolist(),
    )
    save_json(root / "normalization.json", stats)
    return root
