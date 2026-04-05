from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrozenSplit:
    u0: np.ndarray
    uT: np.ndarray


def _load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with (path / "normalization.json").open("r") as f:
        d = json.load(f)
    x_mean = np.asarray(d["x_mean"], dtype=np.float32)
    x_std = np.asarray(d["x_std"], dtype=np.float32)
    y_mean = np.asarray(d["y_mean"], dtype=np.float32)
    y_std = np.asarray(d["y_std"], dtype=np.float32)
    return x_mean, x_std, y_mean, y_std


def load_frozen_split(root: str | Path, split: str, *, D: int | None = None) -> FrozenSplit:
    root = Path(root)
    u0_all = np.load(root / "u0_all.npy")
    uT_all = np.load(root / "uT_all.npy")

    if split == "train":
        if D is None:
            idx = np.load(root / "train_idx.npy")
        else:
            with (root / "train_subsets.json").open("r") as f:
                subsets = json.load(f)
            idx = np.asarray(subsets[str(D)], dtype=int)
    elif split == "val":
        idx = np.load(root / "val_idx.npy")
    elif split == "test":
        idx = np.load(root / "test_idx.npy")
    else:
        raise ValueError(f"Unknown split: {split}")

    return FrozenSplit(u0=u0_all[idx], uT=uT_all[idx])


class FlowMapDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        D: int | None = None,
        normalize: bool = True,
        obs_noise: float = 0.0,
        noise_seed: int = 0,
    ):
        self.root = Path(root)
        self.split = split
        self.D = D
        self.normalize = normalize
        self.obs_noise = obs_noise

        split_data = load_frozen_split(self.root, split, D=D)
        self.u0 = split_data.u0.astype(np.float32)
        self.uT = split_data.uT.astype(np.float32)

        self.x_mean, self.x_std, self.y_mean, self.y_std = _load_normalization(self.root)

        if self.normalize:
            self.u0 = (self.u0 - self.x_mean) / self.x_std
            self.uT = (self.uT - self.y_mean) / self.y_std

        # Add observation noise to training targets (after normalization, reproducible)
        if obs_noise > 0.0 and split == "train":
            rng = np.random.default_rng(noise_seed)
            noise = rng.normal(0, obs_noise, size=self.uT.shape).astype(np.float32)
            self.uT = self.uT + noise

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
