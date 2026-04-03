from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def physics_midpoint_residual(
    *,
    u0: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Implicit midpoint residual using the Lotka–Volterra vector field.

    Shapes: u0, uT_hat: (B, 2)
    Returns: residual (B, 2)
    """
    mid = 0.5 * (uT_hat + u0)
    x = mid[:, 0]
    y = mid[:, 1]

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    f_mid = torch.stack([dx, dy], dim=1)
    return uT_hat - u0 - T * f_mid


def physics_loss(**kwargs) -> torch.Tensor:
    r = physics_midpoint_residual(**kwargs)
    return torch.mean(torch.sum(r**2, dim=1))


def total_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    u0: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    lambda_phys: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ld = mse_loss(pred, target)
    lp = physics_loss(
        u0=u0,
        uT_hat=pred,
        T=T,
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=gamma,
    )
    L = ld + float(lambda_phys) * lp
    return L, {"loss": float(L.detach().cpu()), "data": float(ld.detach().cpu()), "phys": float(lp.detach().cpu())}
