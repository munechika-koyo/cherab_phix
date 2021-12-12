import numpy as np
from matplotlib import pyplot as plt
from .inversion import SVDInversionBase


class GCV(SVDInversionBase):
    """Generalized Cross-Validation (GCV) criterion inversion method inheriting :py:class:`.SVDInversionBase` class.
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
    lambdas : vector-like, optional
        list of regularization parameters to search for optimal one, by default
        ``10 ** np.linspace(-5, 5, 100)``
    **kwargs : :py:class:`.SVDInversionBase` properties, optional
        *kwargs* are used to specify properties like a inversion_base_vectors
    """

    def __init__(self, *args, lambdas=None, **kwargs):

        # initialize originaly valuables
        self._lambda_opt = None
        self._gcvs = None
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
    def gcvs(self):
        """
        :obj:`numpy.ndarray`: list of values of GCV, elements of which are calculated after the optimization iteration
        """
        return self._gcvs

    def optimize(self, itemax=3):
        """excute the optimization of gcv regularization
        In particular, this method is used to search the optimal regularization parameter
        computing gcv function. The optimal regularization parameter corresponds to the minimal
        gcv. This procedure is iterated by up to the itemax times. Every time iterately calculating,
        the range of regularization parameters is narrowed to FWHM around the minimal point.
        The optimal regularization parameter is stored to self._lambda_opt which can be seen :py:attr:`.lambda_opt` property.
        And, lambadas and gcvs is updated and stored to properties.

        Parameters
        ----------
        itemax : int, optional
            iteration times, by default 3
        """
        # preparete local values and caches
        lambdas_cache = self._lambdas.copy()
        lambdas = lambdas_cache.tolist()
        gcvs_cache = np.zeros_like(lambdas)

        # calculate one time
        gcvs_cache = np.array([self.gcv(beta=j) for j in lambdas_cache])
        gcvs = gcvs_cache.tolist()
        # search minimal gcv index
        index_min = np.argmin(gcvs_cache)
        # set property of optimal lambda
        self._lambda_opt = lambdas_cache[index_min]

        # continue to calculate the optimal lambda more than 1 time up to itemax
        if itemax > 1 and isinstance(itemax, int):
            for i in range(itemax - 1):
                # update the range of ragularization parameters
                dlambda_log = np.log10(lambdas_cache[1]) - np.log10(lambdas_cache[0])
                lambda_left_log = np.log10(lambdas_cache[index_min]) - 10 * dlambda_log
                lambda_right_log = np.log10(lambdas_cache[index_min]) + 10 * dlambda_log

                # update lambda's range
                lambdas_cache = 10 ** np.linspace(lambda_left_log, lambda_right_log, 100)
                # compute gcv
                gcvs_cache = np.array([self.gcv(beta=j) for j in lambdas_cache])
                # search maximum gcv index
                index_min = np.argmin(gcvs_cache)
                # set property of optimal lambda
                self._lambda_opt = lambdas_cache[index_min]
                # store caches
                lambdas.extend(lambdas_cache.tolist())
                gcvs.extend(gcvs_cache.tolist())

        # update lambdas and gcvs properties
        lambdas = np.array(lambdas)
        gcvs = np.array(gcvs)
        index_sort = lambdas.argsort()
        self.lambdas = lambdas[index_sort]
        self._gcvs = gcvs[index_sort]
        print(f"completed the optimization (iteration times : {itemax})")

    def gcv(self, beta=None):
        """Computation of GCV criterion function

        Parameters
        ----------
        beta : float, optional
            regularization parameter, by default None

        Returns
        -------
        float
            the value of GCV function at given regularization parameter
        """
        return self.rho(beta) / (1.0 - np.sum(self.w(beta))) ** 2.0

    def optimized_solution(self):
        """calculation inverted solution using GCV criterion optimization

        Returns
        -------
        numpy.ndarray
            optimised inverted solution vector
        """
        if self.lambda_opt is None:
            # excute optimization
            self.optimize()

        # return optimized solution and parameter
        return self.inverted_solution(beta=self.lambda_opt)

    def plot_gcv(self, fig=None, axes=None):
        """Plotting GCV vs regularization parameters in log-log scale

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
        # compute the curvature if self.curvatures doesn't exist.
        if self._gcvs is None:
            self._gcvs = np.array([self.gcv(beta=i) for i in self.lambdas])

        # plotting
        fig = fig or plt.figure()
        ax = axes or fig.add_subplot()
        ax.loglog(self.lambdas, self.gcvs, color="C0")

        # indicate the max point as the optimal point
        if self.lambda_opt is not None:
            self.beta = self.lambda_opt
            ax.scatter(self.beta, self.gcv(beta=self.beta), c="r", marker="x")

        # labels
        ax.set_xlabel("Regularization parameter $\\lambda$")
        ax.set_ylabel("$GCV(\\lambda)$")

        return (fig, ax)
