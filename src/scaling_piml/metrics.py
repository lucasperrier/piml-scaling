from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.vector_norm(pred - target, ord=2, dim=1)
    den = torch.linalg.vector_norm(target, ord=2, dim=1) + eps
    return torch.mean(num / den)
