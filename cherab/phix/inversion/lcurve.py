"""Module for L-curve crietrion."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .inversion import SVDInversionBase

__all__ = ["Lcurve"]


class Lcurve(SVDInversionBase):
    """L curve criterion inversion method inheriting SVDInversionBase class.

    L curve is the trajectory of the point :math:`(\\log||Ax-b||, \\log||L(x-x_0)||)`,
    those of which mean Residual norm and Regularization norm, respectively.
    The "corner" of this curve is assosiated with optimized point of regularization parameter.
    This theory is mentioned by P.C.Hansen.

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
    quiet : bool
        whether or not to show text after finishing the optimization, by default False.
    lambdas : vector_like, optional
        list of regularization parameters to search for optimal one, by default
        ``10 ** numpy.linspace(-5, 5, 100)``
    **kwargs : :py:class:`.SVDInversionBase` properties, optional
        *kwargs* are used to specify properties like a `inversion_base_vectors`
    """

    def __init__(self, *args, lambdas=None, quiet=False, **kwargs):
        # initialize originaly valuables
        self._lambda_opt = 0.0
        self._optimized_solution = np.zeros(0)
        self._curvatures = np.zeros(0)
        self._lambdas = 10 ** np.linspace(-5, 5, 100)
        self._quiet = False

        self.quiet = quiet

        if lambdas is not None:
            self.lambdas = lambdas

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
    def curvatures(self) -> NDArray[np.float64]:
        """list of carvature values.

        This is stored after the optimazation iteration.
        """
        return self._curvatures

    @property
    def quiet(self) -> bool:
        """toggle not to show the text after finishing the optimization."""
        return self._quiet

    @quiet.setter
    def quiet(self, value):
        if not isinstance(value, bool):
            raise TypeError("quiet must be boolen type.")
        else:
            self._quiet = value

    def optimize(self, itemax: int = 5) -> None:
        """Excute the optimization of L-curve regularization. In particular,
        this method is used to seek the optimal regularization parameter
        computing curvature. The optimal regularization parameter corresponds
        to the index of maximum curvature. This procedure is iterated by up to
        the itemax times. Every time iterately calculating, the range of
        regularization parameters is narrowed to FWHM around the maximum
        curvature point. The optimal regularization parameter is cached to
        ``self._lambda_opt`` which can be seen :py:attr:`.lambda_opt` property.
        And, both :py:attr:`.lambdas` and :py:attr:`.curvatures` are updated
        and stored to each property.

        Parameters
        ----------
        itemax
            iteration times, by default 5
        """
        # define local valiables
        lambdas_temp = self._lambdas.copy()
        lambdas = lambdas_temp.tolist()
        curvs_temp = np.zeros_like(lambdas)

        # calculate one time
        curvs_temp = np.array([self.curvature(beta) for beta in lambdas_temp])
        curvs = curvs_temp.tolist()
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
                curvs.extend(curvs_temp.tolist())

        # store lambdas and curvatures as properties
        lambdas = np.array(lambdas)
        curvs = np.array(curvs)
        index_sort = lambdas.argsort()
        self._lambdas = lambdas[index_sort]
        self._curvatures = curvs[index_sort]

        # cache optmized solution
        self._optimized_solution = self.inverted_solution(beta=self._lambda_opt)
        if not self.quiet:
            print(f"completed the optimization (iteration times : {itemax})")

    def optimized_solution(self, itemax: int | None = None) -> NDArray[np.float64]:
        """calculate inverted solution using L-curve criterion optimization.

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

    def plot_L_curve(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
    ) -> tuple[Figure, Axes]:
        """plotting the L curve in log-log scale The range of regularization
        parameters uses self.lambdas.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.
        scatter_plot
            whether or not to plot some L curve points, by default None.
            If you want to manually define the number of points,
            put in the numbers. e.g.) ``scatter_plot=10``.
        scatter_annotate
            whether or not to annotate the scatter_points, by default True.
            This key argument is valid if only ``scatter_plot`` is not None.

        Returns
        ------
        tuple of :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes`
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # compute the norms
        residual_norms = np.array([self.residual_norm(i) for i in self.lambdas])
        regularization_norms = np.array([self.regularization_norm(i) for i in self.lambdas])

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
                np.log10(self.lambdas.min()), np.log10(self.lambdas.max()), scatter_plot
            )
            for _lambda in lambdas:
                point = (self.residual_norm(_lambda), self.regularization_norm(_lambda))
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
                        "$\\lambda$ = {:.2e}".format(_lambda),
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

        # labels
        axes.set_xlabel("Residual norm")
        axes.set_ylabel("Regularization norm")

        return (fig, axes)

    def plot_curvature(
        self, fig: Figure | None = None, axes: Axes | None = None
    ) -> tuple[Figure, Axes]:
        """Plotting L-curve curvature vs regularization parameters.

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
        # compute the curvature
        self._curvatures = np.array([self.curvature(beta) for beta in self.lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plotting
        axes.semilogx(self.lambdas, self.curvatures, color="C0", zorder=0)

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

        lambda_range = (self.lambdas.min(), self.lambdas.max())
        # Draw a y=0 dashed line
        axes.plot(lambda_range, [0, 0], color="k", linestyle="dashed", linewidth=1, zorder=-1)
        # x range limitation
        axes.set_xlim(*lambda_range)
        # labels
        axes.set_xlabel("Regularization parameter $\\lambda$")
        axes.set_ylabel("Curvature of L curve")

        return (fig, axes)

    def curvature(self, beta: float) -> float:
        """calculate curvature for L-curve method L-curve method is used to
        solve the ill-posed inversion equation.

        L-curve is the trajectory of the point :math:`(\\log||Ax - b||, \\log||L(x - x_0)||)`
        varying the regularization parameter `beta`.
        This function returns the value of curvature at one point corresponding to one beta.

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
