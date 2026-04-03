from __future__ import annotations

from dataclasses import asdict
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
