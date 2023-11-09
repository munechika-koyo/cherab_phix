import numpy as np
import pytest
from scipy.sparse import csc_matrix

from cherab.phix.inversion import _SVDBase, compute_svd


@pytest.mark.parametrize("use_gpu", [True, False])
def test_compute_svd(test_data, computed_svd, use_gpu):
    hmat = csc_matrix(np.eye(test_data.matrix.shape[0]))
    s, u, v = compute_svd(test_data.matrix, hmat, use_gpu=use_gpu)

    # compute svd by numpy
    u_np, s_np, vh_np = computed_svd

    # check singular values in the range of matrix rank
    rank = np.linalg.matrix_rank(test_data.matrix)
    np.testing.assert_allclose(s[:rank], s_np[:rank], rtol=0, atol=1.0e-10)


# TODO: in case the matrix is a sparse matrix
def test_compute_svd_sparse():
    pass


@pytest.fixture
def svdbase(test_data, computed_svd):
    u, s, vh = computed_svd
    return _SVDBase(s, u, vh.T, data=test_data.b)


@pytest.fixture
def lambdas():
    return np.logspace(-20, 2, num=500)


class TestSVDBase:
    def test__init(self, test_data, computed_svd):
        u, s, vh = computed_svd
        _SVDBase(s, u, vh.T, data=test_data.b)

    def test_w(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.w(beta)

    def test_rho(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.rho(beta)

    def test_eta(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.eta(beta)

    def test_eta_diff(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.eta_diff(beta)

    def test_residual_norm(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.residual_norm(beta)

    def test_regularization_norm(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.regularization_norm(beta)

    def test_inverted_solution(self, svdbase, lambdas):
        for beta in lambdas:
            svdbase.inverted_solution(beta)


if __name__ == "__main__":
    from cherab.phix.inversion.tests.conftest import TestData

    test_compute_svd(TestData())
