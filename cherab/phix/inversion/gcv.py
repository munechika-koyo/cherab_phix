"""Module for GCV crieterion inversion."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .inversion import SVDInversionBase

__all__ = ["GCV"]


class GCV(SVDInversionBase):
    """Generalized Cross-Validation (GCV) criterion inversion method inheriting
    :obj:`.SVDInversionBase` class.

    GCV criterion function represents as follows:

    .. math::

        GCV(\\lambda) := \\frac{||Ax_\\lambda - b||^2}{\\left[1 - \\sum_{i=1}^N w_i(\\lambda)\\right]^2}

    The optimal regularization paramter is estimated at the minimal of GCV.

    Parameters
    ----------
    s : vector_like
        singular values :math:`\\sigma_i` in :math:`s` vectors.
    u : array_like
        SVD left singular vectors forms as one matrix like :math:`u = (u_1, u_2, ...)`
    vh : array_like
        SVD right singular vectors forms as one matrix like :math:`vh = (v_1, v_2, ...)^T`
    data : vector_like
        given data for inversion calculation
    lambdas : vector_like, optional
        list of regularization parameters to search for optimal one, by default
        ``10 ** numpy.linspace(-5, 5, 100)``
    **kwargs : :py:class:`.SVDInversionBase` properties, optional
        *kwargs* are used to specify properties like a `inversion_base_vectors`
    """

    def __init__(self, *args, lambdas=None, **kwargs):
        # initialize originaly valuables
        self._lambdas = 10 ** np.linspace(-5, 5, 100)
        self._lambda_opt = 0.0
        if lambdas is not None:
            self.lambdas = lambdas
        self._gcvs = np.zeros(0)

        # inheritation
        super().__init__(*args, **kwargs)

    @property
    def lambdas(self) -> NDArray[np.float64]:
        """list of regularization parameters used for the optimization
        process."""
        return self._lambdas

    @lambdas.setter
    def lambdas(self, array):
        if not isinstance(array, np.ndarray) and not isinstance(array, list):
            raise ValueError("lambdas must be the 1-D array of regularization parameters")
        array = np.asarray_chkfinite(array)
        if array.ndim != 1:
            raise ValueError("lambdas must be 1-D array")
        self._lambdas = np.sort(array)

    @property
    def lambda_opt(self) -> float:
        """optimal regularization parameter which is decided after the
        optimization iteration."""
        return self._lambda_opt

    @property
    def gcvs(self) -> NDArray[np.float64]:
        """list of values of GCV, elements of which are calculated after the
        optimization iteration."""
        return self._gcvs

    def optimize(self, itemax: int = 3) -> None:
        """Excute the optimization of L-curve regularization. In particular,
        this method is used to seek the optimal regularization parameter
        computing gcv. The optimal regularization parameter corresponds to the
        index of minimum gcv. This procedure is iterated by up to the itemax
        times. Every time iterately calculating, the range of regularization
        parameters is narrowed to FWHM around the minimum gcv point. The
        optimal regularization parameter is cached to ``self._lambda_opt``
        which can be seen :py:attr:`.lambda_opt` property. And, both
        :py:attr:`.lambdas` and :py:attr:`.gcvs` are updated and stored to each
        property.

        Parameters
        ----------
        itemax
            iteration times, by default 3
        """
        # define local valiables
        lambdas_temp = self._lambdas.copy()
        lambdas = lambdas_temp.tolist()
        gcvs_temp = np.zeros_like(lambdas)

        # calculate one time
        gcvs_temp = np.array([self.gcv(beta=j) for j in lambdas_temp])
        gcvs = gcvs_temp.tolist()
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
                gcvs_temp = np.array([self.gcv(beta=j) for j in lambdas_temp])
                # cache index of minimum gcv
                index_min = np.argmin(gcvs_temp)
                # set property of optimal lambda
                self._lambda_opt = lambdas_temp[index_min]
                # cache temporary lambdas and gcv values
                lambdas.extend(lambdas_temp.tolist())
                gcvs.extend(gcvs_temp.tolist())

        # store lambdas and gcvs properties
        lambdas = np.array(lambdas)
        gcvs = np.array(gcvs)
        index_sort = lambdas.argsort()
        self._lambdas = lambdas[index_sort]
        self._gcvs = gcvs[index_sort]

        # cache optimized solution
        self._optimized_solution = self.inverted_solution(beta=self._lambda_opt)

        print(f"completed the optimization (iteration times : {itemax})")

    def gcv(self, beta: float | None = None) -> float:
        """definition of GCV criterion function.

        Parameters
        ----------
        beta
            regularization parameter, by default None

        Returns
        -------
        float
            the value of GCV function at given regularization parameter
        """
        return self.rho(beta) / (1.0 - np.sum(self.w(beta))) ** 2.0

    def optimized_solution(self, itemax: int | None = None):
        """calculate inverted solution using GCV criterion optimization.

        Parameters
        ----------
        itemax
            iteration times of optimization method, by default None.
            if an integer is given,
            :obj:`.optimize` is called and optimal lambda is stored in :obj:`.lambda_opt`.

        Returns
        -------
        :obj:`numpy.ndarray`
            optimized solution vector
        """
        # excute optimization
        if itemax is not None:
            self.optimize(itemax)
            self._optimized_solution = self.inverted_solution(beta=self.lambda_opt)

        return self._optimized_solution

    def plot_gcv(self, fig: Figure | None = None, axes: Axes | None = None) -> tuple[Figure, Axes]:
        """Plotting GCV vs regularization parameters in log-log scale.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.

        Returns
        ------
        tuple[:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`]
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # compute the curvature if self.curvatures doesn't exist.
        if self._gcvs is None:
            self._gcvs = np.array([self.gcv(beta=i) for i in self.lambdas])

        # plotting
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()
        axes.loglog(self.lambdas, self.gcvs, color="C0")

        # indicate the max point as the optimal point
        if self.lambda_opt is not None:
            self.beta = self.lambda_opt
            axes.scatter(self.beta, self.gcv(beta=self.beta), c="r", marker="x")

        # labels
        axes.set_xlabel("Regularization parameter $\\lambda$")
        axes.set_ylabel("$GCV(\\lambda)$")

        return (fig, axes)
