from __future__ import annotations

import numpy as np


def lotka_volterra_rhs(
    t: float,
    u: np.ndarray,
    *,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
) -> np.ndarray:
    """Lotka–Volterra vector field.

    Args:
        t: time (unused; present for solve_ivp compatibility)
        u: state array shape (2,) or (..., 2) with components [x, y]
    """
    _ = t
    u = np.asarray(u)
    x = u[..., 0]
    y = u[..., 1]

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    return np.stack([dx, dy], axis=-1)
