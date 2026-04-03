import numpy as np

from scaling_piml.systems.lotka_volterra import lotka_volterra_rhs


def test_rhs_shape():
    u = np.array([1.0, 2.0])
    out = lotka_volterra_rhs(0.0, u, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0)
    assert out.shape == (2,)


def test_rhs_batch_shape():
    u = np.array([[1.0, 2.0], [0.5, 1.0]])
    out = lotka_volterra_rhs(0.0, u, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0)
    assert out.shape == (2, 2)
