from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


def solve_flow_map(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    u0: np.ndarray,
    *,
    T: float,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> np.ndarray:
    """Solve the ODE for a single horizon and return u(T).

    Supports u0 shape (2,) for a single IC.
    """
    u0 = np.asarray(u0, dtype=float)
    if u0.shape != (2,):
        raise ValueError(f"Expected u0 shape (2,), got {u0.shape}")

    sol = solve_ivp(rhs, t_span=(0.0, float(T)), y0=u0, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.y[:, -1]


def solve_trajectory(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    u0: np.ndarray,
    *,
    T: float,
    dt: float,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the ODE and return the full trajectory at evenly spaced times.

    Returns (times, states) where times has shape (K,) and states has shape (K, d).
    """
    u0 = np.asarray(u0, dtype=float)
    K = int(round(T / dt)) + 1
    t_eval = np.linspace(0.0, float(T), K)
    sol = solve_ivp(
        rhs, t_span=(0.0, float(T)), y0=u0, t_eval=t_eval,
        method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T
