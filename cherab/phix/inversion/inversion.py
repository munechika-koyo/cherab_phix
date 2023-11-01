"""Module to offer the Base class for various inversion methods."""
from __future__ import annotations

from numpy import asarray, floating, ndarray, sqrt
from numpy.linalg import norm
from scipy.optimize import basinhopping

__all__ = ["_SVDBase"]


class _SVDBase:
    """Base class for inversion calculation based on Singular Value Decomposition (SVD) method.

    .. note::

        This class is designed to be inherited by subclasses which define the objective function
        to optimize the regularization parameter :math:`\\lambda` using the
        :obj:`~scipy.optimize.basinhopping` function.


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
    data : vector_like
        given data for inversion calculation forms as a vector in :math:`\\mathbb{R}^m`

    Notes
    -----
    This class offers the calculation of the inverted solution defined by

    .. math::

        Ax = b,

    where :math:`A` is a matrix in :math:`\\mathbb{R}^{m\\times n}`, :math:`x` is a solution vector
    in :math:`\\mathbb{R}^n` and :math:`b` is a given data vector in :math:`\\mathbb{R}^m`.

    The solution is usually calculated by the least square method, which is defined by

    .. math::

        x_\\text{ls} :&= \\text{argmin} \\{ ||Ax-b||^2 \\} \\

                      &= ( A^\\mathsf{T} A )^{-1} A^\\mathsf{T} b.

    This problem is often ill-posed, so the solution is estimated by adding the regularization term
    like :math:`||Lx||^2` to the right hand side of the equation:

    .. math::

        x_\\lambda :&= \\text{argmin} \\{ ||Ax-b||^2 + \\lambda ||Lx||^2 \\} \\

                    &= (A^\\mathsf{T} A + \\lambda L^\\mathsf{T} L)^{-1} A^\\mathsf{T}\\ b,

    where :math:`\\lambda\\in\\mathbb{R}` is the reguralization parameter and
    :math:`L \\in \\mathbb{R}^{n\\times n}` is a matrix operator in regularization term
    (e.g. laplacian).

    The SVD components are based on the following equation:

    .. math::

        U\\Sigma V^\\mathsf{T}
            = \\begin{pmatrix}
                u_1 & \\cdots & u_r
              \\end{pmatrix}
              \\ \\text{diag}(\\sigma_1,..., \\sigma_r)
              \\ \\begin{pmatrix}
                v_1 & \\cdots & v_r
              \\end{pmatrix}^\\mathsf{T}
            = AL^{-1}

    Using this components allows to reconstruct the estimated solution :math:`x_\\lambda` as follows:

    .. math::

        x_\\lambda &= \\tilde{V}W\\Sigma^{-1}U^\\mathsf{T}b \\\\
                   &= \\begin{pmatrix} \\tilde{v}_1 & \\cdots & \\tilde{v}_r \\end{pmatrix}
                      \\ \\text{diag}(w_1(\\lambda), ..., w_r(\\lambda))
                      \\ \\text{diag}(\\sigma_1^{-1}, ..., \\sigma_r^{-1})
                      \\begin{pmatrix}
                        u_1^\\mathsf{T}b \\\\
                        \\vdots \\\\
                        u_r^\\mathsf{T}b
                      \\end{pmatrix} \\\\
                   &= \\sum_{i=0}^{r} w_i(\\lambda)\\frac{u_i^\\mathsf{T} b}{\\sigma_i} \\tilde{v}_i,

    where :math:`r` is the rank of :math:`A` (:math:`r \\leq \\min(m, n)`), :math:`w_i` is
    the window function, :math:`\\sigma_i` is the singular value of :math:`A` and
    :math:`\\tilde{v}_i` is a :math:`i`-th column vector of the inverted solution basis:
    :math:`\\tilde{V} = L^{-1}V \\in \\mathbb{R}^{n\\times r}`.

    :math:`w_i` is defined as follows:

    .. math::

        w_i(\\lambda) \\equiv \\frac{1}{1 + \\lambda / \\sigma_i^2}.
    """

    def __init__(self, s, u, basis, data=None):
        # validate SVD components
        s = asarray(s, dtype=float)
        if s.ndim != 1:
            raise ValueError("s must be a vector.")

        u = asarray(u, dtype=float)
        if u.ndim != 2:
            raise ValueError("u must be a matrix.")
        if s.size != u.shape[1]:
            raise ValueError("the number of columns of u must be same as that of singular values")

        # set SVD components
        self._s = s
        self._u = u

        # set inverted solution basis
        self.basis = basis

        # set data values
        if data is not None:
            self.data = data
        else:
            self._data = None

        # set initial regularization parameter
        self._beta = 0.0

        # set initial optimal regularization parameter
        self._lambda_opt: float | None = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(s:{self._s.shape}, u:{self._u.shape}, basis:{self._basis.shape})"
        )

    def __getstate__(self):
        """Return the state of the _SVDBase object."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Set the state of the _SVDBase object."""
        self.__dict__.update(state)

    def __reduce__(self):
        return self.__new__, (self.__class__,), self.__getstate__()

    @property
    def s(self) -> ndarray:
        """Singular values of :math:`A`

        Singular values form a vector array like
        :math:`\\sigma = (\\sigma_1, \\sigma_2,...)\\in\\mathbb{R}^r`
        """
        return self._s

    @property
    def u(self) -> ndarray:
        """Left singular vectors of :math:`A`.

        Left singular vactors form a matrix containing column vectors like
        :math:`U = (u_1, u_2,...)\\in\\mathbb{R}^{m\\times r}`
        """
        return self._u

    @property
    def basis(self) -> ndarray:
        """The inverted solution basis :math:`\\tilde{V} \\in \\mathbb{R}^{n\\times r}`.

        If the regularization term is described as :math:`||Lx||^2`, then
        :math:`\\tilde{V} = L^{-1}V \\in \\mathbb{R}^{n\\times r}`,
        where :math:`V\\in\\mathbb{R}^{n\\times r}` is the right singular vectors of :math:`A` and
        :math:`L^{-1}` is the inverse of regularization operator
        :math:`L \\in \\mathbb{R}^{n\\times n}`.
        """
        return self._basis

    @basis.setter
    def basis(self, mat):
        if not isinstance(mat, ndarray):
            raise TypeError("basis must be a numpy.ndarray")
        if mat.shape[1] != self._s.size:
            raise ValueError(
                "the number of columns of inverted solution basis must be same as that of singular values"
            )
        self._basis = mat

    @property
    def data(self) -> ndarray:
        """Given data for inversion calculation."""
        return self._data

    @data.setter
    def data(self, value):
        data = asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._u.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data
        self._ub = self._u.T @ data  # U^T b

    # -------------------------------------------------------------------------
    # Define methods calculating some norms, window function, etc...
    # -------------------------------------------------------------------------

    def w(self, beta: float) -> ndarray:
        """Calculate window function using regularization parameter :math:`\\lambda`.

        The window function is defined as follows:

        .. math::

            w(\\lambda) \\equiv \\frac{1}{1 + \\lambda / \\sigma^2},

        where :math:`\\sigma` is the singular value of :math:`A`.
        Because :math:`\\sigma` is a vector, the window function is also a vector.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        numpy.ndarray (N, )
            vector of window function
        """
        return 1.0 / (1.0 + beta / self._s**2.0)

    def rho(self, beta: float) -> floating:
        """Calculate squared residual norm: :math:`\\rho = ||Ax_\\lambda - b||^2`.

        :math:`\\rho` can be calculated with SVD components as follows:

        .. math::

            \\rho &= \\left\\|
                        U (I_r - W) U^\\mathsf{T} b
                    \\right\\|^2\\\\
                  &= \\left\\|
                        \\begin{pmatrix} u_1 & \\cdots & u_r \\end{pmatrix}
                        (I_r - W) U^\\mathsf{T} b
                     \\right\\|^2\\\\
                  &= \\left\\|
                        (I_r - W) U^\\mathsf{T} b
                     \\right\\|^2
                     \\quad(
                        \\because U^\\mathsf{T}U = I_r,
                        \\quad\\text{i.e.}\\quad u_i\\cdot u_j = \\delta_{ij}
                      ),

        where :math:`W = \\text{diag}(w_1(\\lambda), ..., w_r(\\lambda))`
        and :math:`w_i(\\lambda)` is the window function.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        numpy.floating
            squared residual norm
        """
        return norm((1.0 - self.w(beta)) * self._ub) ** 2.0

    def eta(self, beta: float) -> floating:
        """Calculate squared regularization norm: :math:`\\eta = ||Lx_\\lambda||^2`

        :math:`\\eta` can be calculated with SVD components as follows:

        .. math::

            \\eta &= \\left\\|
                        V W \\Sigma^{-1} U^\\mathsf{T} b
                    \\right\\|^2\\\\
                  &= \\left\\|
                        \\begin{pmatrix} v_1 & \\cdots & v_r \\end{pmatrix}
                        W \\Sigma^{-1} U^\\mathsf{T} b
                     \\right\\|^2\\\\
                  &= \\left\\|
                        W \\Sigma^{-1} U^\\mathsf{T} b
                     \\right\\|^2
                     \\quad(
                        \\because V^\\mathsf{T}V = I_r,
                        \\quad\\text{i.e.}\\quad v_i\\cdot v_j = \\delta_{ij}
                      ),

        where :math:`W = \\text{diag}(w_1(\\lambda), ..., w_r(\\lambda))`
        and :math:`w_i(\\lambda)` is the window function.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        numpy.floating
            squared regularization norm
        """
        return norm((self.w(beta) / self._s) * self._ub) ** 2.0

    def eta_diff(self, beta: float) -> floating:
        """Calculate differential of `eta`: :math:`\\eta' = \\frac{d\\eta}{d\\lambda}`

        Before calculating :math:`\\eta'`, let us calculate the differential of window function
        matrix :math:`W = \\text{diag}(w_1(\\lambda), ..., w_r(\\lambda))` using SVD components:

        .. math::

            \\frac{dW}{d\\lambda}
                &= \\frac{d}{d\\lambda}
                    \\text{diag}\\left(..., \\frac{1}{1 + \\lambda/\\sigma_i^2}, ...\\right)
                    \\quad \\left(\\because w_i(\\lambda) = \\frac{1}{1 + \\lambda/\\sigma_i^2} \\right)\\\\
                &= \\text{diag}\\left(
                    ..., -\\frac{\\sigma_i^{-2}}{(1 + \\lambda/\\sigma_i^2)^2}, ...
                    \\right)\\\\
                &= - W^2 \\Sigma^{-2}\\\\
                &= - \\frac{1}{\\lambda} W (I_r - W). \\quad(\\because I_r - W = \\lambda W \\Sigma^{-2})

        Therefore :math:`\\eta'` can be calculated as follows:

        .. math::

            \\eta' &= \\frac{d}{d\\lambda} \\left\\|W\\Sigma^{^-1}U^\\mathsf{T}b\\right\\|\\\\
                   &= a^\\mathsf{T}\\left(\\frac{d}{d\\lambda} W^2 \\right) a
                        \\quad(\\because a\\equiv\\Sigma^{-1}U^\\mathsf{T}b, \\ W^2 = W^\\mathsf{T}W)\\\\
                   &= 2a^\\mathsf{T}W\\frac{dW}{d\\lambda}a\\\\
                   &= -\\frac{2}{\\lambda} a^\\mathsf{T} W^2 (I_r - W) a\\\\
                   &= -\\frac{2}{\\lambda} a^\\mathsf{T} W^\\mathsf{T} (I_r - W)^{\\mathsf{T}/2}
                        (I_r - W)^{1/2} W a\\\\
                   &= -\\frac{2}{\\lambda}
                        \\left\\|
                            \\sqrt{I_r - W}\\ W \\Sigma^{-1} U^\\mathsf{T} b
                        \\right\\|^2.


        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        numpy.floating
            differential of squared regularization norm
        """
        w = self.w(beta)
        return (-2.0 / beta) * norm(sqrt(1.0 - w) * (w / self._s) * self._ub) ** 2.0

    def residual_norm(self, beta: float) -> ndarray:
        """Return the residual norm: :math:`\\sqrt{\\rho} = ||Ax_\\lambda - b||`

        Parameters
        ----------
        beta
            reguralization parameter

        Returns
        -------
        float
            residual norm
        """
        return sqrt(self.rho(beta))

    def regularization_norm(self, beta: float) -> float:
        """Return the residual norm: :math:`\\sqrt{\\eta} = ||L x_\\lambda||`

        Parameters
        ----------
        beta
            reguralization parameter

        Returns
        -------
        float
            regularization norm
        """
        return sqrt(self.eta(beta))

    # ------------------------------------------------------
    # calculating the inverted solution using SVD components
    # ------------------------------------------------------

    def inverted_solution(self, beta: float) -> ndarray:
        """Calculate the inverted solution using SVD components at given regularization parameter.

        The solution is calculated as follows:

        .. math::

            x_\\lambda
            =
            \\tilde{V}W\\Sigma^{-1}U^\\mathsf{T}b
            =
            \\tilde{V}
            \\begin{pmatrix}
                w_1(\\lambda)\\frac{1}{\\sigma_1} & & \\\\
                & \\ddots & \\\\
                & & w_r(\\lambda)\\frac{1}{\\sigma_r}
            \\end{pmatrix}
            U^\\mathsf{T} b,

        where :math:`\\tilde{V} \\in \\mathbb{R}^{n\\times r}` is the inverted solution basis,
        which is defined by :obj:`.basis` as a property.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        vector_like (N, )
            solution vector
        """
        return self._basis.dot((self.w(beta) / self._s) * self._ub)

    # ------------------------------------------------------
    # Optimization for the regularization parameter
    # ------------------------------------------------------
    @property
    def lambda_opt(self) -> float | None:
        """Optimal regularization parameter defined after `.solve` is executed."""
        return self._lambda_opt

    def solve(
        self,
        bounds: tuple[float, float] = (-20.0, 2.0),
        stepsize: float = 10,
        **kwargs,
    ) -> tuple[ndarray, dict]:
        """Solve the ill-posed inversion equation.

        This method is used to seek the optimal regularization parameter finding the global minimum
        of an objective function using the :obj:`~scipy.optimize.basinhopping` function.

        An objective function `_objective_function` must be defined in the subclass.

        Parameters
        ----------
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
        stepsize
            stepsize of optimization, by default 10.
        **kwargs
            keyword arguments for :obj:`~scipy.optimize.basinhopping` function.

        Returns
        -------
        tuple of :obj:`~numpy.ndarray` and :obj:`~scipy.optimize.OptimizeResult`
            (solution, res), where solution is the inverted solution vector
            and res is the result of optimization generated by :obj:`~scipy.optimize.basinhopping`.
        """
        # initial guess of log10 of regularization parameter
        init_logbeta = 0.5 * (bounds[0] + bounds[1])

        # optimization
        res = basinhopping(
            self._objective_function,
            x0=10**init_logbeta,
            minimizer_kwargs={"bounds": [bounds]},
            stepsize=stepsize,
            **kwargs,
        )

        # set property of optimal lambda
        self._lambda_opt = 10 ** res.x[0]

        # optmized solution
        sol = self.inverted_solution(beta=self._lambda_opt)

        return sol, res

    def _objective_function(self, logbeta: float) -> floating | float:
        raise NotImplementedError("To be defined in subclass.")
