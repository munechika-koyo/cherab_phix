import numpy as np
from matplotlib import pyplot as plt
from .inversion import InversionMethod


class GCV(InversionMethod):
    """Generalized Cross-Validation (GCV) criterion inversion method inheriting InversionMethod class.
    GCV criterion function represents as follows:
    ..:math:
        GCV(\\lambda) := \\frac{||Ax_\\lambda - b||^2}{\\left[1 - \\sum_{i=1}^N w_i(\\lambda)\\right]^2}

    The optimal regularization paramter is estimated at the minimal of GCV.

    Parameters
    -----------
    s : vector_like
        singular values in :math:`s`
    u : array_like
        SVD left singular vectors forms as one matrix like :math:`u = (u_1, u_2, ...)`
    vh : array_like
        SVD right singular vectors forms as one matrix like :math:`vh = (v_1, v_2, ...)^T`
    L_inv : array-like
        inversion matrix in the regularization term. `L_inv` == :math:`L^{-1}` in :math:`||L(x - x_0)||^2`
    data : vector_like
        given data for inversion calculation
    beta : float, optional
        regularization parameter, by default 1.0e-2

    Attributes
    ----------
    optimal : function
        searching optimal lambda
    lambda_opt : float
        optimal lambda
    optimized_solution : function
        returning inverted solultion at the optimal lambda
    plot_gcv
    """

    def __init__(self, s=None, u=None, vh=None, L_inv=None, data=None, beta=None):
        self._lambda_opt = None
        self._gcvs = None
        super().__init__(s=s, u=u, vh=vh, L_inv=L_inv, data=data, beta=beta)

    @property
    def lambda_opt(self):
        return self._lambda_opt

    @property
    def gcvs(self):
        return self._gcvs

    def optimize(self, itemax=3):
        """excute the optimization of gcv regularization
        In particular, this method is used to search the optimal regularization parameter
        computing gcv function. The optimal regularization parameter corresponds to the minimal
        gcv. This procedure is iterated by up to the itemax times. Every time iterately calculating,
        the range of regularization parameters is narrowed to FWHM around the minimal point.
        The optimal regularization parameter is stored to self._lambda_opt which can be seen self.lambda_opt property.
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
        numpy.ndarry
            optimised inverted solution vector
        """
        if self.lambda_opt is None:
            # excute optimization
            self.optimize()

        # return optimized solution and parameter
        return (self.inverted_solution(beta=self.lambda_opt), self.lambda_opt)

    def plot_gcv(self, fig=None, axes=None):
        """Plotting GCV vs regularization parameters in log-log scale

        Parameters
        ----------
        fig : figure object, optional
            matplotlib figure object, by default None
        axes : Axes object, optional
            matplotlib Axes object, by default None

        Returns
        ------
        tuple
            (fig, axes), each of which is matplotlib objects
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
