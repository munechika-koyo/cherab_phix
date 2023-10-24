import numpy as np
import pytest

from cherab.phix.tools.derivative import compute_dmat

# valid cases
CASES = [
    {
        "vmap": np.arange(6).reshape(2, 3),
        "kernel_type": "x",
        "expected": np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 1],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 2),
        "kernel_type": "y",
        "expected": np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 0],
                [0, 0, -1, 0, 1, 0],
                [0, 0, 0, -1, 0, 1],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(2, 1, 3),
        "kernel_type": "laplacian4",
        "expected": np.array(
            [
                [-4, 1, 0, 1, 0, 0],
                [1, -4, 1, 0, 1, 0],
                [0, 1, -4, 0, 0, 1],
                [1, 0, 0, -4, 1, 0],
                [0, 1, 0, 1, -4, 1],
                [0, 0, 1, 0, 1, -4],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 1, 2),
        "kernel_type": "laplacian8",
        "expected": np.array(
            [
                [-8, 1, 1, 1, 0, 0],
                [1, -8, 1, 1, 0, 0],
                [1, 1, -8, 1, 1, 1],
                [1, 1, 1, -8, 1, 1],
                [0, 0, 1, 1, -8, 1],
                [0, 0, 1, 1, 1, -8],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 1, 2),
        "kernel_type": "custom",
        "kernel": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        "expected": np.array(
            [
                [-4, 1, 1, 0, 0, 0],
                [1, -4, 0, 1, 0, 0],
                [1, 0, -4, 1, 1, 0],
                [0, 1, 1, -4, 0, 1],
                [0, 0, 1, 0, -4, 1],
                [0, 0, 0, 1, 1, -4],
            ]
        ),
    },
]

# invalid cases
INVALID_CASES = [
    {
        "vmap": np.zeros((2, 2, 2)),  # invalid shape
        "kernel_type": "x",
        "error": ValueError,
    },
    {
        "vmap": np.zeros((4, 3)),
        "kernel_type": "z",  # invalid kernel type
        "error": ValueError,
    },
    {
        "vmap": np.zeros((3, 5)),
        "kernel_type": "custom",
        "kernel": np.zeros((2, 2, 2)),  # invalid kernel dimension
        "error": ValueError,
    },
    {
        "vmap": np.zeros((5, 5)),
        "kernel_type": "custom",
        "kernel": np.zeros((2, 2)),  # invalid kernel shape
        "error": ValueError,
    },
]


def test_compute_dmat():
    # valid tests
    for case in CASES:
        vmap = case["vmap"]
        kernel_type = case["kernel_type"]
        if kernel_type == "custom":
            kernel = case["kernel"]
            dmat = compute_dmat(vmap, kernel_type=kernel_type, kernel=kernel)
        else:
            dmat = compute_dmat(vmap, kernel_type=kernel_type, kernel=None)
        assert np.allclose(dmat.A, case["expected"])

    # invalid tests
    for case in INVALID_CASES:
        vmap = case["vmap"]
        kernel_type = case["kernel_type"]
        if kernel_type == "custom":
            kernel = case["kernel"]
            with pytest.raises(case["error"]):
                compute_dmat(vmap, kernel_type=kernel_type, kernel=kernel)
        else:
            with pytest.raises(case["error"]):
                compute_dmat(vmap, kernel_type=kernel_type, kernel=None)


if __name__ == "__main__":
    test_compute_dmat()
