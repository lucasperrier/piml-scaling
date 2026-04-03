from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_widths: list[int], activation: str = "relu"):
        super().__init__()
        act = _activation(activation)

        layers: list[nn.Module] = []
        prev = in_dim
        for w in hidden_widths:
            layers.append(nn.Linear(prev, w))
            layers.append(act)
            prev = w
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


CAPACITY_GRID: dict[str, list[int]] = {
    "tiny": [32, 32],
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
    "xlarge": [256, 256, 256],
}
