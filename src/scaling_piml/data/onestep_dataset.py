"""One-step dataset: extracts (x_t, x_{t+Δt}) pairs from dense trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OneStepDataset(Dataset):
    """Dataset of one-step pairs (state_t, state_{t+Δt}) from dense trajectories.

    D controls the number of *trajectories* used (not the total number of pairs),
    keeping the D interpretation consistent with FlowMapDataset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        dt: float,
        D: int | None = None,
        normalize: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.dt = dt
        self.D = D
        self.normalize = normalize

        # Load trajectory data
        trajectories = np.load(self.root / "trajectories.npy")  # (N_total, K, 2)
        times = np.load(self.root / "times.npy")  # (K,)

        # Load metadata to get dt_fine
        with (self.root / "metadata.json").open() as f:
            meta = json.load(f)
        dt_fine = meta["dt_fine"]

        # Compute stride
        stride = int(round(dt / dt_fine))
        if stride < 1:
            raise ValueError(f"dt={dt} is smaller than dt_fine={dt_fine}")
        actual_dt = stride * dt_fine
        if abs(actual_dt - dt) > 1e-10:
            raise ValueError(
                f"dt={dt} is not an integer multiple of dt_fine={dt_fine} "
                f"(closest is {actual_dt})"
            )

        # Load split indices
        if split == "train":
            if D is not None:
                with (self.root / "train_subsets.json").open() as f:
                    subsets = json.load(f)
                traj_idx = np.asarray(subsets[str(D)], dtype=int)
            else:
                traj_idx = np.load(self.root / "train_idx.npy")
        elif split == "val":
            traj_idx = np.load(self.root / "val_idx.npy")
        elif split == "test":
            traj_idx = np.load(self.root / "test_idx.npy")
        else:
            raise ValueError(f"Unknown split: {split}")

        # Extract consecutive pairs at the requested stride
        selected = trajectories[traj_idx]  # (N_traj, K, 2)
        K = selected.shape[1]
        n_pairs_per_traj = (K - 1) // stride

        # Build flat arrays of (input, target) pairs
        inputs = []
        targets = []
        for k in range(n_pairs_per_traj):
            t_start = k * stride
            t_end = t_start + stride
            if t_end >= K:
                break
            inputs.append(selected[:, t_start, :])
            targets.append(selected[:, t_end, :])

        self.u0 = np.concatenate(inputs, axis=0).astype(np.float32)  # (N_pairs, 2)
        self.uT = np.concatenate(targets, axis=0).astype(np.float32)

        # Load normalization stats
        with (self.root / "normalization.json").open() as f:
            norm = json.load(f)
        self.x_mean = np.asarray(norm["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(norm["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(norm["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(norm["y_std"], dtype=np.float32)

        if self.normalize:
            self.u0 = (self.u0 - self.x_mean) / self.x_std
            self.uT = (self.uT - self.y_mean) / self.y_std

    def denormalize_inputs(self, values: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return values
        mean = torch.as_tensor(self.x_mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.x_std, device=values.device, dtype=values.dtype)
        return values * std + mean

    def denormalize_targets(self, values: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return values
        mean = torch.as_tensor(self.y_mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.y_std, device=values.device, dtype=values.dtype)
        return values * std + mean

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.u0[idx])
        y = torch.from_numpy(self.uT[idx])
        return x, y
