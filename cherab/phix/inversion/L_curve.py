import numpy as np
from matplotlib import pyplot as plt
from .inversion import SVDInversionBase


class Lcurve(SVDInversionBase):
    """L curve criterion inversion method inheriting SVDInversionBase class.
    L curve is the trajectory of the point :math:`(\\log||Ax-b||, \\log||L(x-x_0)||)`, those of which
    mean Residual norm and Regularization norm, respectively.
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
    lambdas : vector-like, optional
        list of regularization parameters to search for optimal one, by default
        `10 ** np.linspace(-5, 5, 100)`
    **kwargs : :py:class:`.SVDInversionBase` properties, optional
        *kwargs* are used to specify properties like a inversion_base_vectors
    """

    def __init__(self, *args, lambdas=None, **kwargs):

        # initialize originaly valuables
        self._lambda_opt = None
        self._curvatures = None
        self._lambdas = 10 ** np.linspace(-5, 5, 100)

        if lambdas:
            self.lambdas = lambdas

        # inheritation
        super().__init__(*args, **kwargs)

    @property
    def lambdas(self):
        """
        :obj:`numpy.ndarray`: list of regularization parameters used for the optimization process
        """
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
    def lambda_opt(self):
        """
        float: optimal regularization parameter which is decided after the optimization iteration
        """
        return self._lambda_opt

    @property
    def curvatures(self):
        """
        :obj:`numpy.ndarray`: list of carvature values.
        This is stored after the optimazation iteration.
        """
        return self._curvatures

    def optimize(self, itemax=5):
        """excute the optimization of L-curve regularization
        In particular, this method is used to search the optimal regularization parameter
        computing curvature. The optimal regularization parameter corresponds to the maximum
        curvature. This procedure is iterated by up to the itemax times. Every time iterately calculating,
        the range of regularization parameters is narrowed to FWHM around the maximum point.
        The optimal regularization parameter is stored to self._lambda_opt which can be seen :py:attr:`.lambda_opt` property.
        And, lambadas and curvatures is updated and stored to properties.

        Parameters
        ----------
        itemax : int, optional
            iteration times, by default 5
        """
        # preparete local values and caches
        lambdas_cache = self._lambdas.copy()
        lambdas = lambdas_cache.tolist()
        curvs_cache = np.zeros_like(lambdas)

        # calculate one time
        curvs_cache = np.array([curvature(self.rho(j), self.eta(j), self.eta_diff(j), beta=j) for j in lambdas_cache])
        curvs = curvs_cache.tolist()
        # search maximum curvature index
        index_max = np.argmax(curvs_cache)
        # set property of optimal lambda
        self._lambda_opt = lambdas_cache[index_max]

        # continue to calculate the optimal lambda more than 1 time up to itemax
        if itemax > 1 and isinstance(itemax, int):
            for i in range(itemax - 1):
                # update the range of ragularization parameters

                # check if curvature has positive values
                if curvs_cache.max() > 0:
                    # if curvs.max() > 0, the range of that is narrowd within FWHM
                    half = curvs_cache.max() * 0.5
                    if half < curvs_cache.min():
                        half = (curvs_cache.max() + curvs_cache.min()) * 0.5
                    signs = np.sign(curvs_cache - half)
                    zero_crossings = signs[0:-2] != signs[1:-1]
                    # search nearest neighbor point of peak point
                    zero_crossings_i = np.where(zero_crossings)[0]
                    zero_crossings_near = np.abs(np.where(zero_crossings)[0] - index_max).argmin()
                    zero_crossings_i = zero_crossings_i[zero_crossings_near]

                    # calculate FWFH in logscale
                    fwfh_log = np.abs(np.log10(lambdas_cache[zero_crossings_i]) - np.log10(self._lambda_opt))
                    # if zero_crossings_i is leftside of peak
                    if zero_crossings_i < index_max:
                        lambda_left_log = np.log10(lambdas_cache[zero_crossings_i])
                        lambda_right_log = np.log10(self._lambda_opt) + fwfh_log
                    else:
                        lambda_left_log = np.log10(self._lambda_opt) - fwfh_log
                        lambda_right_log = np.log10(lambdas_cache[zero_crossings_i])

                else:
                    # if curvature does not have any positive values, the range of lambdas is expanded to both sides.
                    dlambda_log = np.log10(lambdas_cache[1]) - np.log10(lambdas_cache[0])
                    lambda_left_log = np.log10(lambdas_cache[0]) - 100 * dlambda_log
                    lambda_right_log = np.log10(lambdas_cache[-1]) + 100 * dlambda_log

                # update lambda's range
                lambdas_cache = 10 ** np.linspace(lambda_left_log, lambda_right_log, 100)
                # compute curvature
                curvs_cache = np.array(
                    [curvature(self.rho(j), self.eta(j), self.eta_diff(j), beta=j) for j in lambdas_cache]
                )
                # search maximum curvature index
                index_max = np.argmax(curvs_cache)
                # set property of optimal lambda
                self._lambda_opt = lambdas_cache[index_max]
                # store caches
                lambdas.extend(lambdas_cache.tolist())
                curvs.extend(curvs_cache.tolist())

        # update lambdas and curvatures properties
        lambdas = np.array(lambdas)
        curvs = np.array(curvs)
        index_sort = lambdas.argsort()
        self.lambdas = lambdas[index_sort]
        self._curvatures = curvs[index_sort]
        print(f"completed the optimization (iteration times : {itemax})")

    def optimized_solution(self, itemax=5):
        """calculation inverted solution using L-curve criterion optimization

        Parameters
        ----------
        itemax : int, optional
            iteration times of optimization method, by default 5

        Returns
        -------
        numpy.ndarray
            optimised inverted solution vector
        """
        # excute optimization
        self.optimize(itemax)

        # return optimized solution and parameter
        return self.inverted_solution(beta=self.lambda_opt)

    def plot_L_curve(self, fig=None, axes=None, scatter_plot="off", scatter_annotate=True):
        """plotting the L curve in log-log scale
        The range of regularization parameters uses self.lambdas

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            matplotlib figure object, by default None.
        axes : matplotlib.axes.Axes, optional
            matplotlib Axes object, by default None.
        scatter_plot : str or int, optional
            whether or not to plot some L curve points,
            by default "off". if yes and you want to manually define the number of points,
            put in the numbers. For example, scatter_plot=10.
        scatter_annotate : bool, optional
            whether or not to annotate the scatter_points, by default True.
            This key argument is valid if only `scatter_plot` is not "off".

        Returns
        ------
        tuple of :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes`
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # compute the norms
        residual_norms = np.array([self.residual_norm(i) for i in self.lambdas])
        regularization_norms = np.array([self.regularization_norm(i) for i in self.lambdas])

        # plotting
        fig = fig or plt.figure()
        ax = axes or fig.add_subplot()
        ax.loglog(residual_norms, regularization_norms, color="C0")

        # plot some points of L curve and annotate with regularization parameters label
        if isinstance(scatter_plot, int):
            lambdas = 10 ** np.linspace(np.log10(self.lambdas.min()), np.log10(self.lambdas.max()), scatter_plot)
            for _lambda in lambdas:
                point = (self.residual_norm(_lambda), self.regularization_norm(_lambda))
                ax.scatter(point[0], point[1], edgecolors="C0", marker="o", facecolor="none")
                if scatter_annotate is True:
                    ax.annotate("$\\lambda$ = {:.2e}".format(_lambda), xy=point, color="k")

        # plot L curve corner if already optimize method excuted
        if self.lambda_opt is not None:
            ax.scatter(
                self.residual_norm(self.lambda_opt), self.regularization_norm(self.lambda_opt), c="r", marker="x"
            )

        # labels
        ax.set_xlabel("Residual norm")
        ax.set_ylabel("Regularization norm")

        return (fig, ax)

    def plot_curvature(self, fig=None, axes=None):
        """Plotting L-curve curvature vs regularization parameters

        Parameters
        ----------
        fig : :obj:`matplotlib.figure.Figure`, optional
            matplotlib figure object, by default None.
        axes : :obj:`matplotlib.axes.Axes`, optional
            matplotlib Axes object, by default None.

        Returns
        ------
        tuple of :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes`
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # compute the curvature
        self._curvatures = np.array(
            [curvature(self.rho(i), self.eta(i), self.eta_diff(i), beta=i) for i in self.lambdas]
        )

        # plotting
        fig = fig or plt.figure()
        ax = axes or fig.add_subplot()
        ax.semilogx(self.lambdas, self.curvatures, color="C0")

        # indicate the max point as the optimal point
        if self.lambda_opt is not None:
            self.beta = self.lambda_opt
            ax.scatter(self.beta, curvature(self.rho(), self.eta(), self.eta_diff(), beta=self.beta), c="r", marker="x")

        lambda_range = (self.lambdas.min(), self.lambdas.max())
        # Draw a y=0 dashed line
        ax.plot(lambda_range, [0, 0], color="k", linestyle="dashed", linewidth=1)
        # x range limitation
        ax.set_xlim(*lambda_range)
        # labels
        ax.set_xlabel("Regularization parameter $\\lambda$")
        ax.set_ylabel("Curvature of L curve")

        return (fig, ax)


def curvature(rho=None, eta=None, eta_dif=None, beta=1.0e-2):
    """calculate curvature for L-curve method
    L-curve method is used to solve the ill-posed inversion equation.
    L-curve is the trajectory of the point :math:`(\\log||Ax - b||, \\log||L(x - x_0)||)`
    varying the regularization parameter `beta`.
    This function returns the value of curvature at one point corresponding to one beta.

    Parameters
    ----------
    rho : float, required
        residual of least squared :math:`\\rho = ||Ax - b||^2`
    eta : float, required
        squared norm of regularization term :math:`\\eta = ||L(x - x_0)||^2`
    eta_dif : float, required
        differencial of `eta` :math:`\\frac{d \\eta}{d \\lambda}`
    beta : float, optional
        regularization parameter, by default 1.0e-2

    Returns
    -------
    float
        the value of calculated curvature
    """
    return (
        -2.0
        * rho
        * eta
        * (eta * beta ** 2.0 + beta * rho + rho * eta / eta_dif)
        / ((beta * eta) ** 2.0 + rho ** 2.0) ** 1.5
    )
