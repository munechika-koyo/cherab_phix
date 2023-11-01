import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cherab.phix.inversion.gcv import GCV


class TestGCV:
    def test_gcv(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        gcv = GCV(sigma, u, vh.T, data=test_data.b)

        # try to compute gcv with some log10(lambda) values
        logbetas = np.linspace(-20, 2, num=500)
        for logbeta in logbetas:
            gcv.gcv(logbeta)

    def test__objective_function(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        gcv = GCV(sigma, u, vh.T, data=test_data.b)

        # try to compute objective function with some log10(lambda) values
        logbetas = np.linspace(-20, 2, num=500)
        for logbeta in logbetas:
            gcv._objective_function(logbeta)

    def test_solve(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        gcv = GCV(sigma, u, vh.T, data=test_data.b)

        bounds = (-20.0, 2.0)
        stepsize = 10

        sol, status = gcv.solve(bounds=bounds, stepsize=stepsize, disp=True)

        # TODO: check the solution
        # this test is not passed because the gcv optimization is not converged at this ill-posed
        # problem. we need to find a better test case.

    def test_plot_gcv(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        gcv = GCV(sigma, u, vh.T, data=test_data.b)

        bounds = (-20.0, 2.0)
        n_beta = 500

        fig, ax = gcv.plot_gcv(bounds=bounds, n_beta=n_beta)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
