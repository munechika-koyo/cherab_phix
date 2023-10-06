"""Module to offer the Base class for various inversion methods."""
from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

__all__ = ["SVDInversionBase"]


class SVDInversionBase:
    """Base class for inversion calculation based on singular value decomposition (SVD) method.

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
    like :math:`||L(x - x_0)||^2` to the right hand side of the equation:

    .. math::

        x_\\lambda :&= \\text{argmin} \\{ ||Ax-b||^2 + \\lambda ||L(x - x_0)||^2 \\} \\

                    &= (A^\\mathsf{T} A + \\lambda L^\\mathsf{T} L)^{-1}
                        (A^\\mathsf{T}\\ b + \\lambda L^\\mathsf{T} Lx_0),

    where :math:`\\lambda\\in\\mathbb{R}` is the reguralization parameter,
    :math:`L \\in \\mathbb{R}^{n\\times n}` is a matrix operator in regularization term
    (e.g. laplacian) and :math:`x_0\\in\\mathbb{R}^n` is a prior assumption.

    The SVD components are based on the following equation:

    .. math::

        U\\Sigma V^\\mathsf{T}
            = (u_1, u_2, ...)
              \\cdot
              \\text{diag}(\\sigma_1, \\sigma_2,...)
              \\cdot
              (v_1, v_2, ...)^\\mathsf{T}
            = AL^{-1}

    Using this components allows to reconstruct the estimated solution :math:`x_\\lambda` as follows:

    .. math::

        x_\\lambda = \\sum_{i=0}^{r} w_i(\\lambda)\\frac{u_i \\cdot b}{\\sigma_i} \\tilde{v}_i,

    where :math:`r` is the rank of :math:`A` (:math:`r \\leq \\min(m, n)`), :math:`w_i` is
    the window function, :math:`\\sigma_i` is the singular value of :math:`A` and
    :math:`\\tilde{v}_i` is a :math:`i`-th column vector of the inverted solution basis:
    :math:`\\tilde{V} = L^{-1}V \\in \\mathbb{R}^{n\\times r}`.

    :math:`w_i` is defined as follows:

    .. math::

        w_i(\\lambda) \\equiv \\frac{1}{1 + \\lambda / \\sigma_i^2}.

    Parameters
    ----------
    s : vector_like
        singular values of :math:`A`
        like :math:`\\sigma = (\\sigma_1, \\sigma_2, ...) \\in \\mathbb{R}^r`
    u : array_like
        left singular vectors of :math:`A`
        like :math:`U = (u_1, u_2, ...) \\in \\mathbb{R}^{m\\times r}`
    basis
        inverted solution basis :math:`\\tilde{V} \\in \\mathbb{R}^{n\\times r}`.
        Here, :math:`\\tilde{V} = L^{-1}V`, where :math:`V\\in\\mathbb{R}^{n\\times r}` is
        the right singular vectors of :math:`A` and :math:`L^{-1}` is the inverse of
        regularization operator :math:`L \\in \\mathbb{R}^{n\\times n}`.
    data : vector_like
        given data for inversion calculation forms as a vector in :math:`\\mathbb{R}^m`
    """

    def __init__(self, s, u, basis: NDArray[float64] | csr_matrix, data=None):
        # validate SVD components
        s = np.asarray_chkfinite(s, dtype=float)
        if s.ndim != 1:
            raise ValueError("s must be a vector.")

        u = np.asarray_chkfinite(u, dtype=float)
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
        self.data = data

        # set initial regularization parameter
        self._beta = 0.0

    @property
    def s(self) -> NDArray[float64]:
        """singular values of :math:`A` like
        :math:`\\sigma = (\\sigma_1, \\sigma_2, ...) \\in \\mathbb{R}^r`
        """
        return self._s

    @property
    def u(self) -> NDArray[float64]:
        """left singular vectors of :math:`A`
        like :math:`U = (u_1, u_2, ...) \\in \\mathbb{R}^{m\\times r}`
        """
        return self._u

    @property
    def basis(self) -> NDArray[float64] | csr_matrix:
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
        if not isinstance(mat, (np.ndarray, csr_matrix)):
            raise TypeError("basis must be a numpy.ndarray or scipy.sparse.csr_matrix")
        if mat.shape[1] != self._s.size:
            raise ValueError(
                "the number of columns of inverted solution basis must be same as that of singular values"
            )
        self._basis = mat

    @property
    def data(self) -> NDArray[float64]:
        """Given data for inversion calculation."""
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray_chkfinite(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._u.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data
        self._ub = self._u.T.dot(self._data)  # U^T b

    # -------------------------------------------------------------------------
    # Define methods calculating some norms, window function, etc...
    # -------------------------------------------------------------------------

    def w(self, beta: float) -> NDArray[float64]:
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

    def rho(self, beta: float) -> np.floating:
        """Calculate squared residual norm: :math:`\\rho = ||Ax_\\lambda - b||^2`.

        :math:`\\rho` can be calculated with SVD components as follows:

        .. math::

            \\rho = \\left\\|
                        (1 - w(\\lambda)) \\cdot U^\\mathsf{T}b
                    \\right\\|^2,

        where :math:`w(\\lambda)` is a vector of the window function.

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

    def eta(self, beta: float) -> np.floating:
        """Calculate squared regularization norm: :math:`\\eta = ||L(x_\\lambda - x_0)||^2`

        :math:`\\eta` can be calculated with SVD components as follows:

        .. math::

            \\eta = \\left\\|
                \\begin{pmatrix}
                    w_1(\\lambda)/\\sigma_1\\\\
                    \\vdots \\\\
                    w_r(\\lambda)/\\sigma_r
                \\end{pmatrix}
                \\cdot
                U^\\mathsf{T}b
            \\right\\|^2,

        where :math:`w_i(\\lambda)` is the window function.

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

    def eta_diff(self, beta: float) -> np.floating:
        """Calculate differential of `eta`: :math:`\\eta' = \\frac{d\\eta}{d\\lambda}`

        :math:`\\eta'` can be calculated with SVD components as follows:

        .. math::

            \\eta' = -\\frac{2}{\\lambda}
                \\left\\|
                    \\sqrt{1 - w(\\lambda)}
                    \\cdot
                    \\begin{pmatrix}
                        w_1(\\lambda)/\\sigma_1\\\\
                        \\vdots \\\\
                        w_r(\\lambda)/\\sigma_r
                    \\end{pmatrix}
                    \\cdot
                    U^\\mathsf{T}b
                \\right\\|^2,

        where :math:`w(\\lambda)` is a vector of the window function and
        :math:`w_i(\\lambda)` is the window function.

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
        return (-2.0 / beta) * norm(np.sqrt(1.0 - w) * (w / self._s) * self._ub) ** 2.0

    def residual_norm(self, beta: float) -> NDArray[float64]:
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
        return np.sqrt(self.rho(beta))

    def regularization_norm(self, beta: float) -> float:
        """Return the residual norm: :math:`\\sqrt{\\eta} = ||L (x_\\lambda - x_0)||`

        Parameters
        ----------
        beta
            reguralization parameter

        Returns
        -------
        float
            regularization norm
        """
        return np.sqrt(self.eta(beta))

    # ------------------------------------------------------
    # calculating the inverted solution using SVD components
    # ------------------------------------------------------

    def inverted_solution(self, beta: float) -> NDArray[float64]:
        """Calculate the inverted solution using SVD components at given regularization parameter.

        The solution is calculated as follows:

        .. math::

            x_\\lambda = \\tilde{V}
            \\begin{pmatrix}
                w_1(\\lambda) / \\sigma_1\\\\
                \\vdots \\\\
                w_r(\\lambda) / \\sigma_r
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
        numpy.ndarray (N, )
            solution vector
        """
        return self._basis.dot((self.w(beta) / self._s) * self._ub)
