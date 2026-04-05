from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


# ---------------------------------------------------------------------------
# Conservation-law prior: H(u,v) = δu − γ ln u + βv − α ln v
# ---------------------------------------------------------------------------

def _lv_invariant(
    u: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Evaluate the Lotka–Volterra first integral H(u,v).

    u: (B, 2)  with columns [prey, predator].
    Returns: (B,)
    """
    x = u[:, 0]
    y = u[:, 1]
    return delta * x - gamma * torch.log(x) + beta * y - alpha * torch.log(y)


def conservation_loss(
    *,
    u0: torch.Tensor,
    uT_hat: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Mean squared violation of the Lotka–Volterra conservation law."""
    H0 = _lv_invariant(u0, alpha=alpha, beta=beta, delta=delta, gamma=gamma)
    HT = _lv_invariant(uT_hat, alpha=alpha, beta=beta, delta=delta, gamma=gamma)
    return torch.mean((HT - H0) ** 2)


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


# ---------------------------------------------------------------------------
# Composite 2-step midpoint ODE residual (halved step size, ~4x lower bias)
#
# The model outputs (u_{T/2}, u_T) ∈ R^4.  The residual applies the midpoint
# rule on each half-interval [0, T/2] and [T/2, T] separately.
# ---------------------------------------------------------------------------

def _lv_vector_field(
    u: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Evaluate the Lotka–Volterra vector field F(u).

    u: (B, 2)  → returns (B, 2)
    """
    x = u[:, 0]
    y = u[:, 1]
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return torch.stack([dx, dy], dim=1)


def composite_midpoint_loss(
    *,
    u0: torch.Tensor,
    uT2_hat: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Composite 2-step midpoint ODE residual loss.

    Applies midpoint-rule residual on [0, T/2] and [T/2, T]:
        r_a = u_{T/2} - u_0 - (T/2) F((u_0 + u_{T/2})/2)
        r_b = u_T - u_{T/2} - (T/2) F((u_{T/2} + u_T)/2)
        L = mean(|r_a|^2 + |r_b|^2)

    On ground truth at T=1.0 this has ~24x lower loss than single-step midpoint.
    """
    h = T / 2.0
    kwargs = dict(alpha=alpha, beta=beta, delta=delta, gamma=gamma)

    mid_a = 0.5 * (u0 + uT2_hat)
    r_a = uT2_hat - u0 - h * _lv_vector_field(mid_a, **kwargs)

    mid_b = 0.5 * (uT2_hat + uT_hat)
    r_b = uT_hat - uT2_hat - h * _lv_vector_field(mid_b, **kwargs)

    return torch.mean(torch.sum(r_a**2, dim=1)) + torch.mean(torch.sum(r_b**2, dim=1))


def total_loss_composite(
    *,
    pred_full: torch.Tensor,
    pred_target: torch.Tensor,
    target: torch.Tensor,
    u0: torch.Tensor,
    uT2_hat_phys: torch.Tensor,
    uT_hat_phys: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    lambda_phys: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss for the composite 2-step midpoint model.

    pred_full: (B, 4) raw model output in normalized space
    pred_target: (B, 2) last-2 components (the u_T prediction) in normalized space
    target: (B, 2) ground-truth u_T in normalized space
    """
    ld = mse_loss(pred_target, target)
    lp = composite_midpoint_loss(
        u0=u0,
        uT2_hat=uT2_hat_phys,
        uT_hat=uT_hat_phys,
        T=T,
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=gamma,
    )
    L = ld + float(lambda_phys) * lp
    return L, {
        "loss": float(L.detach().cpu()),
        "data": float(ld.detach().cpu()),
        "phys": float(lp.detach().cpu()),
    }


# ---------------------------------------------------------------------------
# Simpson's 1/3-rule ODE residual (4th-order, true quadrature)
#
# The model outputs (u_{T/2}, u_T) ∈ R^4 (same layout as composite).
# Residual: r = u_T - u_0 - (T/6)[F(u_0) + 4F(u_{T/2}) + F(u_T)]
# ---------------------------------------------------------------------------

def simpson_residual(
    *,
    u0: torch.Tensor,
    uT2_hat: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Simpson's 1/3-rule ODE residual (4th-order).

    Shapes: u0, uT2_hat, uT_hat: (B, 2)
    Returns: residual (B, 2)
    """
    kwargs = dict(alpha=alpha, beta=beta, delta=delta, gamma=gamma)

    F0 = _lv_vector_field(u0, **kwargs)
    F_mid = _lv_vector_field(uT2_hat, **kwargs)
    F_T = _lv_vector_field(uT_hat, **kwargs)

    return uT_hat - u0 - (T / 6.0) * (F0 + 4.0 * F_mid + F_T)


def simpson_loss(
    *,
    u0: torch.Tensor,
    uT2_hat: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> torch.Tensor:
    """Mean squared Simpson's 1/3-rule ODE residual loss."""
    r = simpson_residual(
        u0=u0, uT2_hat=uT2_hat, uT_hat=uT_hat,
        T=T, alpha=alpha, beta=beta, delta=delta, gamma=gamma,
    )
    return torch.mean(torch.sum(r**2, dim=1))


def total_loss_simpson(
    *,
    pred_full: torch.Tensor,
    pred_target: torch.Tensor,
    target: torch.Tensor,
    u0: torch.Tensor,
    uT2_hat_phys: torch.Tensor,
    uT_hat_phys: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    lambda_phys: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss for the true Simpson's 1/3-rule model.

    Same interface as total_loss_composite (4D output: u_{T/2}, u_T).
    """
    ld = mse_loss(pred_target, target)
    lp = simpson_loss(
        u0=u0,
        uT2_hat=uT2_hat_phys,
        uT_hat=uT_hat_phys,
        T=T,
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=gamma,
    )
    L = ld + float(lambda_phys) * lp
    return L, {
        "loss": float(L.detach().cpu()),
        "data": float(ld.detach().cpu()),
        "phys": float(lp.detach().cpu()),
    }


def total_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    u0: torch.Tensor,
    uT_hat_phys: torch.Tensor,
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
        uT_hat=uT_hat_phys,
        T=T,
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=gamma,
    )
    L = ld + float(lambda_phys) * lp
    return L, {
        "loss": float(L.detach().cpu()),
        "data": float(ld.detach().cpu()),
        "phys": float(lp.detach().cpu()),
    }


def total_loss_conservation(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    u0: torch.Tensor,
    uT_hat_phys: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    lambda_phys: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ld = mse_loss(pred, target)
    lp = conservation_loss(
        u0=u0,
        uT_hat=uT_hat_phys,
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma=gamma,
    )
    L = ld + float(lambda_phys) * lp
    return L, {
        "loss": float(L.detach().cpu()),
        "data": float(ld.detach().cpu()),
        "phys": float(lp.detach().cpu()),
    }


# ===========================================================================
# Duffing oscillator losses
# ===========================================================================

def _duffing_vector_field(
    u: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Evaluate the Duffing oscillator vector field F(u).

    u: (B, 2).  System: dx/dt = y, dy/dt = -alpha*x - beta*x^3
    Returns: (B, 2)
    """
    x = u[:, 0]
    y = u[:, 1]
    dx = y
    dy = -alpha * x - beta * x**3
    return torch.stack([dx, dy], dim=1)


def _duffing_energy(
    u: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Duffing energy: E = 1/2 y^2 + 1/2 alpha x^2 + 1/4 beta x^4.

    u: (B, 2).  Returns: (B,)
    """
    x = u[:, 0]
    y = u[:, 1]
    return 0.5 * y**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4


def duffing_midpoint_residual(
    *,
    u0: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Implicit midpoint residual for the Duffing oscillator."""
    mid = 0.5 * (uT_hat + u0)
    f_mid = _duffing_vector_field(mid, alpha=alpha, beta=beta)
    return uT_hat - u0 - T * f_mid


def duffing_physics_loss(
    *,
    u0: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    r = duffing_midpoint_residual(u0=u0, uT_hat=uT_hat, T=T, alpha=alpha, beta=beta)
    return torch.mean(torch.sum(r**2, dim=1))


def duffing_composite_midpoint_loss(
    *,
    u0: torch.Tensor,
    uT2_hat: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Composite 2-step midpoint ODE residual for Duffing."""
    h = T / 2.0
    kw = dict(alpha=alpha, beta=beta)

    mid_a = 0.5 * (u0 + uT2_hat)
    r_a = uT2_hat - u0 - h * _duffing_vector_field(mid_a, **kw)

    mid_b = 0.5 * (uT2_hat + uT_hat)
    r_b = uT_hat - uT2_hat - h * _duffing_vector_field(mid_b, **kw)

    return torch.mean(torch.sum(r_a**2, dim=1)) + torch.mean(torch.sum(r_b**2, dim=1))


def duffing_conservation_loss(
    *,
    u0: torch.Tensor,
    uT_hat: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Mean squared violation of the Duffing energy conservation."""
    E0 = _duffing_energy(u0, alpha=alpha, beta=beta)
    ET = _duffing_energy(uT_hat, alpha=alpha, beta=beta)
    return torch.mean((ET - E0) ** 2)


def duffing_simpson_loss(
    *,
    u0: torch.Tensor,
    uT2_hat: torch.Tensor,
    uT_hat: torch.Tensor,
    T: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Simpson's 1/3-rule ODE residual loss for the Duffing oscillator (4th-order)."""
    kw = dict(alpha=alpha, beta=beta)
    F0 = _duffing_vector_field(u0, **kw)
    F_mid = _duffing_vector_field(uT2_hat, **kw)
    F_T = _duffing_vector_field(uT_hat, **kw)
    r = uT_hat - u0 - (T / 6.0) * (F0 + 4.0 * F_mid + F_T)
    return torch.mean(torch.sum(r**2, dim=1))
