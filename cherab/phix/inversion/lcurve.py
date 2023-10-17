"""Module for L-curve crietrion."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .inversion import _SVDBase

__all__ = ["Lcurve"]


class Lcurve(_SVDBase):
    """L-curve criterion optimization for regularization parameter.

    L curve is the trajectory of the point :math:`(\\log||Ax_\\lambda-b||, \\log||L(x_\\lambda-x_0)||)`,
    those of which mean the residual and the regularization norm, respectively.
    The "corner" of this curve is assosiated with optimized point of regularization parameter, where
    the curvature of the L curve is maximized.
    This theory is mentioned by P.C.Hansen [1]_.

    Parameters
    ----------
    s : vector_like
        singular values of :math:`A`
        like :math:`\\sigma = (\\sigma_1, \\sigma_2, ...) \\in \\mathbb{R}^r`
    u : array_like
        left singular vectors of :math:`A`
        like :math:`U = (u_1, u_2, ...) \\in \\mathbb{R}^{m\\times r}`
    basis : array_like
        inverted solution basis :math:`\\tilde{V} \\in \\mathbb{R}^{n\\times r}`.
        Here, :math:`\\tilde{V} = L^{-1}V`, where :math:`V\\in\\mathbb{R}^{n\\times r}` is
        the right singular vectors of :math:`A` and :math:`L^{-1}` is the inverse of
        regularization operator :math:`L \\in \\mathbb{R}^{n\\times n}`.
    **kwargs : :py:class:`._SVDBase` properties, optional
        *kwargs* are used to specify properties like a `data`

    References
    ----------
    .. [1] P.C.Hansen, "The L-curve and its use in the numerical treatment of inverse problems",
           January 2001 In book: Computational Inverse Problems in Electrocardiology,
           Publisher: WIT Press
    """

    def __init__(self, *args, **kwargs):
        # initialize originaly valuables
        self._lambdas = None

        # inheritation
        super().__init__(*args, **kwargs)

    def optimize(
        self, itemax: int = 5, bounds: tuple[float, float] = (-20.0, 2.0)
    ) -> NDArray[np.float64]:
        """Excute the optimization of L-curve regularization.

        Warnings
        --------
        This method will be deprecated in the future. Please use :py:meth:`.solve` instead.

        This method is used to seek the optimal regularization parameter computing curvature.
        The optimal regularization parameter corresponds to the index of maximum curvature.
        This procedure is iterated by up to the `itemax` times. Every time iterately calculating,
        the range of regularization parameters is narrowed to FWHM around the maximum curvature
        point. The optimal regularization parameter is cached to ``self._lambda_opt``
        which can be seen :py:attr:`.lambda_opt` property.
        And, :py:attr:`.lambdas` is updated and stored to the property.

        Parameters
        ----------
        itemax
            iteration times, by default 5
        bounds
            initial bounds of log10 of regularization parameter, by default (-20.0, 2.0).

        Returns
        -------
        :obj:`numpy.ndarray`
            optimized solution vector
        """

        # define regularization parameters from log10 of bounds
        lambdas_temp = np.logspace(*bounds, 100)

        # cache lambdas as list
        lambdas = lambdas_temp.tolist()

        # calculate one time
        curvs_temp = np.array([self.curvature(beta) for beta in lambdas_temp])

        # cache index of maximum curvature
        index_max = np.argmax(curvs_temp)

        # set property of optimal lambda
        self._lambda_opt = lambdas_temp[index_max]

        # continue to seek the optimal lambda more than 1 time up to itemax
        if isinstance(itemax, int) and itemax > 1:
            for _ in range(itemax - 1):
                # check if curvature has positive values
                if curvs_temp.max() > 0:
                    # the range of lambdas is narrowd within FWHM
                    half = curvs_temp.max() * 0.5
                    if half < curvs_temp.min():
                        half = (curvs_temp.max() + curvs_temp.min()) * 0.5
                    signs = np.sign(curvs_temp - half)
                    zero_crossings = signs[0:-2] != signs[1:-1]
                    # search nearest neighbor point of peak point
                    zero_crossings_i = np.where(zero_crossings)[0]
                    zero_crossings_near = np.abs(zero_crossings_i - index_max).argmin()
                    zero_crossings_i = zero_crossings_i[zero_crossings_near]

                    # calculate FWHM in logscale
                    fwhm_log = np.abs(
                        np.log10(lambdas_temp[zero_crossings_i]) - np.log10(self._lambda_opt)
                    )
                    # if zero_crossings_i is leftside of peak
                    if zero_crossings_i < index_max:
                        lambda_left_log = np.log10(lambdas_temp[zero_crossings_i])
                        lambda_right_log = np.log10(self._lambda_opt) + fwhm_log
                    else:
                        lambda_left_log = np.log10(self._lambda_opt) - fwhm_log
                        lambda_right_log = np.log10(lambdas_temp[zero_crossings_i])

                else:
                    # if curvature does not have any positive values, the range of lambdas is expanded to both sides.
                    dlambda_log = np.log10(lambdas_temp[1]) - np.log10(lambdas_temp[0])
                    lambda_left_log = np.log10(lambdas_temp[0]) - 100 * dlambda_log
                    lambda_right_log = np.log10(lambdas_temp[-1]) + 100 * dlambda_log

                # update the range of lambdas
                lambdas_temp = 10 ** np.linspace(lambda_left_log, lambda_right_log, 100)
                # compute curvature
                curvs_temp = np.array([self.curvature(beta) for beta in lambdas_temp])
                # cache index of maximum curvature
                index_max = np.argmax(curvs_temp)
                # set property of optimal lambda
                self._lambda_opt = lambdas_temp[index_max]
                # cache temporary lambdas and cavature values
                lambdas.extend(lambdas_temp.tolist())

        # cache lambdas as property
        lambdas = np.array(lambdas)
        self._lambdas = lambdas.sort()

        return self.inverted_solution(self._lambda_opt)

    def plot_L_curve(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: tuple[float, float] = (-20.0, 2.0),
        n_beta: int = 100,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plotting the L curve in log-log scale.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
            This is not used if :obj:`.lambda` is not None.
        n_beta
            number of regularization parameters, by default 100.
            This is not used if :obj:`.lambda` is not None.
        scatter_plot
            whether or not to plot some L curve points, by default None.
            If you want to manually define the number of points,
            put in the numbers. e.g.) ``scatter_plot=10``.
        scatter_annotate
            whether or not to annotate the scatter_points, by default True.
            This key argument is valid if only ``scatter_plot`` is not None.

        Returns
        -------
        tuple of :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes`
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # define regularization parameters
        if self._lambdas is None:
            lambdas = np.logspace(*bounds, n_beta)
        else:
            lambdas = self._lambdas

        # compute norms
        residual_norms = np.array([self.residual_norm(i) for i in lambdas])
        regularization_norms = np.array([self.regularization_norm(i) for i in lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plotting
        axes.loglog(residual_norms, regularization_norms, color="C0", zorder=0)

        # plot some points of L curve and annotate with regularization parameters label
        if isinstance(scatter_plot, int) and scatter_plot > 0:
            lambdas = 10 ** np.linspace(
                np.log10(lambdas.min()), np.log10(lambdas.max()), scatter_plot
            )
            for beta in lambdas:
                point = (self.residual_norm(beta), self.regularization_norm(beta))
                axes.scatter(
                    point[0],
                    point[1],
                    edgecolors="C0",
                    marker="o",
                    facecolor="none",
                    zorder=1,
                )
                if scatter_annotate is True:
                    axes.annotate(
                        "$\\lambda$ = {:.2e}".format(beta),
                        xy=point,
                        color="k",
                        zorder=2,
                    )

        # plot L curve corner if already optimize method excuted
        if self.lambda_opt is not None:
            axes.scatter(
                self.residual_norm(self.lambda_opt),
                self.regularization_norm(self.lambda_opt),
                c="r",
                marker="x",
                zorder=2,
                label=f"$\\lambda = {self.lambda_opt:.2e}$",
            )
            axes.legend()

        # labels
        axes.set_xlabel("Residual norm")
        axes.set_ylabel("Regularization norm")

        return (fig, axes)

    def plot_curvature(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: tuple[float, float] = (-20.0, 2.0),
        n_beta: int = 100,
    ) -> tuple[Figure, Axes]:
        """Plotting L-curve curvature vs regularization parameters.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
            This is not used if :obj:`.lambda` is not None.
        n_beta
            number of regularization parameters, by default 100.
            This is not used if :obj:`.lambda` is not None.

        Returns
        -------
        tuple[:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`]
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # define regularization parameters
        if self._lambdas is None:
            lambdas = np.logspace(*bounds, n_beta)
        else:
            lambdas = self._lambdas

        # compute the curvature
        curvatures = np.array([self.curvature(beta) for beta in lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plotting
        axes.semilogx(lambdas, curvatures, color="C0", zorder=0)

        # indicate the maximum point as the optimal point
        if self.lambda_opt is not None:
            axes.scatter(
                self.lambda_opt,
                self.curvature(self.lambda_opt),
                c="r",
                marker="x",
                zorder=1,
                label=f"$\\lambda = {self.lambda_opt:.2e}$",
            )

        lambda_range = (lambdas.min(), lambdas.max())

        # Draw a y=0 dashed line
        axes.plot(lambda_range, [0, 0], color="k", linestyle="dashed", linewidth=1, zorder=-1)

        # x range limitation
        axes.set_xlim(*lambda_range)

        # labels
        axes.set_xlabel("Regularization parameter $\\lambda$")
        axes.set_ylabel("Curvature of L curve")

        return (fig, axes)

    def curvature(self, beta: float) -> float:
        """Calculate L-curve curvature.

        This method calculates the L-curve curvature :math:`\\kappa` specified by a regularization
        parameter :math:`\\beta` as follows:

        .. math::

            \\begin{align}
                \\kappa(\\beta) &= \\frac{f^{\\prime\\prime}(x)}{\\left[1 + f^{\\prime}(x)^2\\right]^{3/2}}
                                 = -2 \\eta\\rho
                                    \\frac{\\beta^2 \\eta + \\beta \\rho + \\rho\\eta/\\eta^\\prime}
                                          {(\\beta^2 \\eta^2 + \\rho^2)^{3/2}},\\\\
                \\rho &\\equiv ||Ax_\\beta - b||_2^2,\\\\
                \\eta &\\equiv ||L(x_\\beta - x_0)||_2^2,\\\\
                \\eta^\\prime &\\equiv \\frac{d\\eta}{d\\beta}.
            \\end{align}

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        float
            the value of calculated curvature
        """
        rho = self.rho(beta)
        eta = self.eta(beta)
        eta_dif = self.eta_diff(beta)

        return (
            -2.0
            * rho
            * eta
            * (eta * beta**2.0 + beta * rho + rho * eta / eta_dif)
            / ((beta * eta) ** 2.0 + rho**2.0) ** 1.5
        )

    def _objective_function(self, logbeta: float) -> float:
        """Objective function for optimization.

        The optimal regularization parameter corresponds to the index of maximum curvature.
        So, this function is defined as the negative value of curvature.

        Parameters
        ----------
        logbeta
            log10 of regularization parameter

        Returns
        -------
        float
            negative value of curvature
        """
        return -self.curvature(10**logbeta)
