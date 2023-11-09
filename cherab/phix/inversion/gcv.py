"""Module for GCV crieterion inversion."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .inversion import _SVDBase

__all__ = ["GCV"]


class GCV(_SVDBase):
    """Generalized Cross-Validation (GCV) criterion optimization for regularization parameter.

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

    Notes
    -----
    GCV criterion function is defined as follows:

    .. math::

        GCV(\\lambda) := \\frac{||Ax_\\lambda - b||^2}{\\left[1 - \\sum_{i=1}^N w_i(\\lambda)\\right]^2}

    The optimal regularization parameter corresponds to the minimum value of GCV function [1]_.

    References
    ----------
    .. [1] IWAMA Naofumi and OHDACHI Satoshi, "Numerical Methods of Tomography for Plasma
        Diagnostics", Journal of Plasma and Fusion Research, Vol.82, No.7 (2006) 399 - 409
    """

    def __init__(self, *args, **kwargs):
        # initialize originaly valuables
        self._lambdas = None

        # inheritation
        super().__init__(*args, **kwargs)

    def optimize(
        self, itemax: int = 5, bounds: tuple[float, float] = (-20.0, 2.0)
    ) -> NDArray[np.float64]:
        """Optimize the regularization parameter using GCV criterion.

        Warnings
        --------
        This method will be deprecated in the future. Please use :py:meth:`.solve` instead.

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
        gcvs_temp = np.array([self.gcv(beta) for beta in lambdas_temp])

        # cache index of minimum gcv
        index_min = np.argmin(gcvs_temp)

        # set property of optimal lambda
        self._lambda_opt = lambdas_temp[index_min]

        # continue to seek the optimal lambda more than 1 time up to itemax
        if isinstance(itemax, int) and itemax > 1:
            for _ in range(itemax - 1):
                # TODO: implement FWHM calculation
                # define left and right edge of lambda
                dlambda_log = np.log10(lambdas_temp[1]) - np.log10(lambdas_temp[0])
                lambda_left_log = np.log10(lambdas_temp[index_min]) - 10 * dlambda_log
                lambda_right_log = np.log10(lambdas_temp[index_min]) + 10 * dlambda_log

                # update the range of lambdas
                lambdas_temp = 10 ** np.linspace(lambda_left_log, lambda_right_log, 100)
                # calculate gcv
                gcvs_temp = np.array([self.gcv(beta) for beta in lambdas_temp])
                # cache index of minimum gcv
                index_min = np.argmin(gcvs_temp)
                # set property of optimal lambda
                self._lambda_opt = lambdas_temp[index_min]
                # cache temporary lambdas and gcv values
                lambdas.extend(lambdas_temp.tolist())

        # store lambdas and gcvs properties
        lambdas = np.array(lambdas)
        self._lambdas = lambdas.sort()

        return self.inverted_solution(self._lambda_opt)

    def gcv(self, beta: float) -> np.floating:
        """Calculate of GCV criterion function.

        GCV can be calculated as follows:

        .. math::

            GCV(\\lambda) = \\frac{\\rho}{\\left[1 - \\sum_{i=1}^r w_i(\\lambda)\\right]^2},

        where :math:`\\rho` is the squared residual norm and :math:`w_i(\\lambda)` is the
        window function.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        float
            the value of GCV function at given regularization parameter
        """
        return self.rho(beta) / (1.0 - np.sum(self.w(beta))) ** 2.0

    def _objective_function(self, logbeta: float) -> np.floating:
        """Objective function for optimization.

        The optimal regularization parameter corresponds to the minimum value of GCV function.

        Parameters
        ----------
        logbeta
            log10 of regularization parameter

        Returns
        -------
        float
            the value of GCV function at given regularization parameter
        """
        return self.gcv(10**logbeta)

    def plot_gcv(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: tuple[float, float] = (-20.0, 2.0),
        n_beta: int = 100,
    ) -> tuple[Figure, Axes]:
        """Plotting GCV vs regularization parameters in log-log scale.

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

        # calculate GCV values
        gcvs = np.array([self.gcv(beta) for beta in lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plot
        axes.loglog(lambdas, gcvs, color="C0", zorder=0)

        # indicate the max point as the optimal point
        if self.lambda_opt is not None:
            axes.scatter(
                self.lambda_opt,
                self.gcv(self.lambda_opt),
                c="r",
                marker="x",
                zorder=1,
                label=f"$\\lambda = {self.lambda_opt:.2e}$",
            )

        # x range limitation
        axes.set_xlim(lambdas.min(), lambdas.max())

        # labels
        axes.set_xlabel("Regularization parameter $\\lambda$")
        axes.set_ylabel("$GCV(\\lambda)$")

        return (fig, axes)
