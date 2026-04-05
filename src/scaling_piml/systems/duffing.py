from __future__ import annotations

import numpy as np


def duffing_rhs(
    t: float,
    u: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Unforced undamped Duffing oscillator vector field.

    System:
        dx/dt = y
        dy/dt = -alpha * x - beta * x^3

    Args:
        t: time (unused; present for solve_ivp compatibility)
        u: state array shape (2,) or (..., 2) with components [x, y]
        alpha: linear stiffness (default 1.0)
        beta: cubic stiffness (default 1.0, hardening spring)
    """
    _ = t
    u = np.asarray(u)
    x = u[..., 0]
    y = u[..., 1]

    dx = y
    dy = -alpha * x - beta * x**3

    return np.stack([dx, dy], axis=-1)


def duffing_energy(
    u: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Duffing energy (conserved for undamped, unforced system).

    E = (1/2) y^2 + (1/2) alpha x^2 + (1/4) beta x^4
    """
    u = np.asarray(u)
    x = u[..., 0]
    y = u[..., 1]
    return 0.5 * y**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4
