"""Generate dense trajectory datasets for one-step task experiments.

For each IC, solves the full trajectory at fine dt and saves:
  - trajectories.npy  (N_total, K, d)
  - times.npy         (K,)
  - metadata.json     (dt_fine, T, system, ...)
  - train_idx.npy, val_idx.npy, test_idx.npy
  - train_subsets.json
  - normalization.json

Usage:
    python scripts/generate_trajectory_datasets.py \
        --config configs/default.yaml \
        --system lotka-volterra \
        --seeds 11,22,33 \
        --T 5.0 --dt-fine 0.001 \
        --out data-trajectories/lotka-volterra
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scaling_piml.config_loader import load_experiment_config
from scaling_piml.data.generate import sample_initial_conditions
from scaling_piml.systems.lotka_volterra import lotka_volterra_rhs
from scaling_piml.systems.duffing import duffing_rhs
from scaling_piml.systems.van_der_pol import van_der_pol_rhs
from scaling_piml.systems.ode import solve_trajectory
from scaling_piml.utils.io import ensure_dir, save_json
from scaling_piml.utils.seed import seed_everything


def _make_rhs(system_name: str, system_cfg):
    if system_name == "duffing":
        def rhs(t, u):
            return duffing_rhs(t, u, alpha=system_cfg.alpha, beta=system_cfg.beta)
    elif system_name == "van-der-pol":
        def rhs(t, u):
            return van_der_pol_rhs(t, u, mu=system_cfg.mu)
    else:
        def rhs(t, u):
            return lotka_volterra_rhs(
                t, u,
                alpha=system_cfg.alpha, beta=system_cfg.beta,
                delta=system_cfg.delta, gamma=system_cfg.gamma,
            )
    return rhs


def generate_for_seed(
    *,
    data_seed: int,
    out_dir: Path,
    cfg,
    system_name: str,
    T: float,
    dt_fine: float,
) -> Path:
    seed_everything(data_seed)
    rng = np.random.default_rng(data_seed)

    root = ensure_dir(out_dir / f"data_seed={data_seed}")

    n_total = cfg.data.train_pool + cfg.data.val_size + cfg.data.test_size
    u0_all = sample_initial_conditions(cfg.data, n_total, rng=rng)

    rhs = _make_rhs(system_name, cfg.system)
    K = int(round(T / dt_fine)) + 1

    trajectories = np.zeros((n_total, K, 2), dtype=np.float64)
    times = None

    for i in range(n_total):
        if i % 500 == 0:
            print(f"  [seed={data_seed}] Solving trajectory {i}/{n_total}")
        t, states = solve_trajectory(
            rhs, u0_all[i], T=T, dt=dt_fine,
            method=cfg.solver.method, rtol=cfg.solver.rtol, atol=cfg.solver.atol,
        )
        trajectories[i] = states
        if times is None:
            times = t

    # Split indices (same logic as generate.py)
    idx = np.arange(n_total)
    rng.shuffle(idx)

    train_idx = idx[:cfg.data.train_pool]
    val_idx = idx[cfg.data.train_pool:cfg.data.train_pool + cfg.data.val_size]
    test_idx = idx[cfg.data.train_pool + cfg.data.val_size:]

    # Normalization: computed from ALL states in training trajectories
    train_states = trajectories[train_idx].reshape(-1, 2)
    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = (train_states.std(axis=0) + 1e-12).astype(np.float32)

    # Save everything
    np.save(root / "trajectories.npy", trajectories.astype(np.float32))
    np.save(root / "times.npy", times.astype(np.float64))
    np.save(root / "train_idx.npy", train_idx)
    np.save(root / "val_idx.npy", val_idx)
    np.save(root / "test_idx.npy", test_idx)

    # Nested subsets (first D of train_idx, same as flow-map)
    subsets = {str(D): train_idx[:D].tolist() for D in cfg.data.dataset_sizes}
    save_json(root / "train_subsets.json", subsets)

    # Normalization (same format as FlowMapDataset expects)
    norm = {
        "x_mean": state_mean.tolist(),
        "x_std": state_std.tolist(),
        "y_mean": state_mean.tolist(),
        "y_std": state_std.tolist(),
    }
    save_json(root / "normalization.json", norm)

    # Metadata
    meta = {
        "system": system_name,
        "T": T,
        "dt_fine": dt_fine,
        "K": K,
        "n_total": n_total,
        "train_pool": cfg.data.train_pool,
        "val_size": cfg.data.val_size,
        "test_size": cfg.data.test_size,
        "data_seed": data_seed,
    }
    save_json(root / "metadata.json", meta)

    print(f"  Saved {root}  ({n_total} trajectories, K={K})")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--system", type=str, default="lotka-volterra",
                        choices=["lotka-volterra", "duffing", "van-der-pol"])
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt-fine", type=float, default=0.001)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.system:
        cfg.system.name = args.system

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    out_dir = Path(args.out)

    for seed in seeds:
        print(f"Generating trajectory data for seed={seed}...")
        generate_for_seed(
            data_seed=seed,
            out_dir=out_dir,
            cfg=cfg,
            system_name=args.system,
            T=args.T,
            dt_fine=args.dt_fine,
        )

    print("Done.")


if __name__ == "__main__":
    main()
