import numpy as np
import pytest


def true_func(x):
    """True function."""
    return 2.0 * np.exp(-6.0 * (x - 0.8) ** 2) + np.exp(-2.0 * (x + 0.5) ** 2)


def kernel(s, t):
    """Kernel function."""
    u = np.pi * (np.sin(s) + np.sin(t))
    if u == 0:
        return np.cos(s) + np.cos(t)
    else:
        return (np.cos(s) + np.cos(t)) * (np.sin(u) / u) ** 2


class TestData:
    """Test data class offering a simple ill-posed inverse problem."""

    def __init__(self):
        _t = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=100)
        _s = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=100)

        # construct operator matrix
        matrix = np.array([[kernel(i, j) for j in _t] for i in _s])
        matrix[:, 0] *= 0.5
        matrix[:, -1] *= 0.5
        matrix *= np.abs(_t[1] - _t[0])

        # set attributes
        self.matrix: np.ndarray = matrix
        self.x_true: np.ndarray = true_func(_t)

        # mesured exact unperturbed data and added white noise
        b_0 = matrix.dot(self.x_true)
        rng = np.random.default_rng()
        b_noise = rng.normal(0, 1.0e-4, b_0.size)
        self.b = b_0 + b_noise


@pytest.fixture
def test_data():
    return TestData()


@pytest.fixture
def computed_svd(test_data):
    u, sigma, vh = np.linalg.svd(test_data.matrix, full_matrices=False)
    return u, sigma, vh
