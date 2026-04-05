from __future__ import annotations

import numpy as np


def van_der_pol_rhs(
    t: float,
    u: np.ndarray,
    *,
    mu: float = 1.0,
) -> np.ndarray:
    """Van der Pol oscillator vector field.

    System:
        dx/dt = y
        dy/dt = mu * (1 - x^2) * y - x

    Args:
        t: time (unused; present for solve_ivp compatibility)
        u: state array shape (2,) or (..., 2) with components [x, y]
        mu: nonlinearity parameter (default 1.0; limit cycle exists for mu > 0)
    """
    _ = t
    u = np.asarray(u)
    x = u[..., 0]
    y = u[..., 1]

    dx = y
    dy = mu * (1 - x**2) * y - x

    return np.stack([dx, dy], axis=-1)
