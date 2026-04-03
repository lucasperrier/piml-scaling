import numpy as np

from scaling_piml.systems.lotka_volterra import lotka_volterra_rhs
from scaling_piml.systems.ode import solve_flow_map


def test_solve_flow_map_runs():
    def rhs(t, u):
        return lotka_volterra_rhs(t, u, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0)

    uT = solve_flow_map(rhs, np.array([1.0, 1.0]), T=1.0)
    assert uT.shape == (2,)
    assert np.isfinite(uT).all()
