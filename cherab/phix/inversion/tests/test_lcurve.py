import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cherab.phix.inversion import Lcurve


class TestLcurve:
    def test_carvature(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        lcurve = Lcurve(sigma, u, vh.T, data=test_data.b)

        # try to compute curvature with some log10(lambda) values
        logbetas = np.linspace(-20, 2, num=500)
        for logbeta in logbetas:
            lcurve.curvature(logbeta)

    def test__objective_function(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        lcurve = Lcurve(sigma, u, vh.T, data=test_data.b)

        # try to compute objective function with some log10(lambda) values
        logbetas = np.linspace(-20, 2, num=500)
        for logbeta in logbetas:
            lcurve._objective_function(logbeta)

    def test_solve(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        lcurve = Lcurve(sigma, u, vh.T, data=test_data.b)

        bounds = (-20.0, 2.0)
        stepsize = 10

        sol, status = lcurve.solve(bounds=bounds, stepsize=stepsize, disp=True)

        assert status["success"] is True
        np.testing.assert_allclose(sol, test_data.x_true, rtol=0, atol=1.0)

    @pytest.mark.parametrize(("scatter_plot", "scatter_annotate"), [(None, False), (5, True)])
    def test_plot_L_curve(self, test_data, computed_svd, scatter_plot, scatter_annotate):
        u, sigma, vh = computed_svd
        lcurve = Lcurve(sigma, u, vh.T, data=test_data.b)

        bounds = (-20.0, 2.0)
        n_beta = 500

        fig, ax = lcurve.plot_L_curve(
            bounds=bounds,
            n_beta=n_beta,
            scatter_plot=scatter_plot,
            scatter_annotate=scatter_annotate,
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_curvature(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        lcurve = Lcurve(sigma, u, vh.T, data=test_data.b)

        bounds = (-20.0, 2.0)
        n_beta = 500

        fig, ax = lcurve.plot_curvature(bounds=bounds, n_beta=n_beta)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
