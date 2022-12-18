"""Module to offer the Base class for various inversion methods."""
from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.linalg import norm
from numpy.typing import NDArray

__all__ = ["SVDInversionBase"]


class SVDInversionBase:
    """Base class for inversion calculation based on singular value
    decomposition (eco algorithum i.e. not full matrices of `u`, `vh`).

    This provides users useful tools for regularization computation using SVD components.
    The estimated solution :math:`x_\\lambda` is defined by the following linear equation:

    .. math::

        Ax = b,

    Where :math:`A` is a :math:`M \\times N` matrix. Adding the regularization term,
    the estimated solution :math:`x_\\lambda` is derived from the following equation:

    .. math::

        x_\\lambda :&= \\text{argmin} \\{ ||Ax-b||^2 + \\lambda ||L(x - x_0)||^2 \\} \\

                   &= ( A^\\intercal A + \\lambda L^\\intercal L )^{-1} (A^\\intercal\\ b + \\lambda L^\\intercal Lx_0),

    Where :math:`\\lambda` is the reguralization parameter, :math:`L` is a matrix operator in regularization term (e.g. laplacian)
    and :math:`x_0` is a prior assumption.

    The SVD components are based on the following equation:

    .. math::

        U\\Sigma V^\\intercal = (u_1, u_2, ...) \\cdot \\text{diag}(\\sigma_1, \\sigma_2,...) \\cdot (v_1, v_2, ...)^\\intercal = AL^{-1}

    Using this components, The :math:`x_\\lambda` can be reconstructed as follows:

    .. math::

        x_\\lambda = \\sum_{i=0}^{K} w_i(\\lambda)\\frac{u_i \\cdot b}{\\sigma_i} L^{-1} v_i, \\quad (K \\equiv \\min(M, N)),

    :math:`w_i` is defined as follows:

    .. math::

        w_i \\equiv \\frac{1}{1 + \\lambda / \\sigma_i^2}.

    Parameters
    -----------
    s : vector_like
        singular values :math:`\\sigma_i` in :math:`s` vectors.
    u : array_like
        SVD left singular vectors forms as one matrix like :math:`u = (u_1, u_2, ...)`
    vh : array_like
        SVD right singular vectors forms as one matrix like :math:`vh = (v_1, v_2, ...)^T`
    data : vector_like
        given data for inversion calculation
    inversion_base_vectors : array-like, optional
        The components of inversions base represented as ``L_inv @ vh.T``.
        This property is offered to speed up the calculation of inversions.
        If None, it is automatically computed when calculating the inverted solution.
    L_inv : array_like, optional
        inversion matrix in the regularization term. :obj:`L_inv` is :math:`L^{-1}` in :math:`||L(x - x_0)||^2`
    """

    def __init__(self, s, u, vh, data, inversion_base_vectors=None, L_inv=None):
        # set SVD values
        self._s = s
        self._u = u
        self._vh = vh

        # inversion base
        self._inversion_base_vectors = None
        if inversion_base_vectors is not None:
            self.inversion_base_vectors = inversion_base_vectors

        # set matrix in the regularization term
        self._L_inv = None
        if L_inv is not None:
            self.L_inv = L_inv

        # set data values
        self.data = data

        # set initial regularization parameter
        self._beta = 0.0

    @property
    def s(self) -> NDArray[float64]:
        """
        vector_like: singular values :math:`\\sigma_i` in :math:`s` vectors.
        """
        return self._s

    @property
    def u(self) -> NDArray[float64]:
        """
        array_like: SVD left singular vectors forms as one matrix like :math:`u = (u_1, u_2, ...)`
        """
        return self._u

    @property
    def vh(self) -> NDArray[float64]:
        """
        array_like: SVD right singular vectors forms as one matrix like :math:`vh = (v_1, v_2, ...)^T`
        """
        return self._vh

    @property
    def L_inv(self) -> NDArray[float64] | None:
        """inversion matrix in the regularization term.

        :obj:`L_inv` is :math:`L^{-1}` in :math:`||L(x - x_0)||^2`,
        by default ``numpy.identity(self._vh.shape[1])``
        """
        return self._L_inv

    @L_inv.setter
    def L_inv(self, value):
        inv_mat = np.asarray(value)
        n, m = inv_mat.shape
        if m != self._vh.shape[1] and n == m:
            raise ValueError("L_inv must be a square.")
        self._L_inv = inv_mat

    @property
    def inversion_base_vectors(self) -> NDArray[float64] | None:
        """
        array_like or None if not set: The components of inversions base represented as ``L_inv @ vh.T``.
        This property is offered to speed up the calculation of inversions.
        If None, it is automatically computed when calculating the inverted solution.
        """
        return self._inversion_base_vectors

    @inversion_base_vectors.setter
    def inversion_base_vectors(self, mat):
        mat = np.asarray_chkfinite(mat)
        if mat.shape[1] != self._s.size:
            raise ValueError(
                "the number of columns of Image Base matrix must be same as the one of singular values"
            )
        self._inversion_base_vectors = mat

    @property
    def data(self) -> NDArray[float64]:
        """given data for inversion calculation."""
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray_chkfinite(value, dtype=float).ravel()
        if data.size != self._u.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data
        self._ub = np.dot(np.transpose(self._u), self._data)

    @property
    def beta(self) -> float:
        """regularization parameter."""
        return self._beta

    @beta.setter
    def beta(self, value):
        if not isinstance(value, float):
            raise ValueError("regularization parameter beta must be one float number.")
        self._beta = value

    # -------------------------------------------------------------------------
    # Define methods calculating some norms, window function, etc...
    # -------------------------------------------------------------------------

    def w(self, beta: float | None = None) -> NDArray[float64]:
        """calculate window function using regularization parameter as a
        valuable and using singular values.

        Parameters
        -----------
        beta
            regularization parameter, by default None

        Returns
        --------
        numpy.ndarray (N, )
            window function :math:`\\frac{1}{1 + \\lambda / \\sigma_i^2}`
        """
        if beta is None:
            beta = self._beta
        return 1.0 / (1.0 + beta / self._s**2.0)

    def rho(self, beta: float | None = None) -> np.floating:
        """
        calculate squared residual norm :math:`\\rho = ||Ax - b||^2`.

        Parameters
        ----------
        beta
            regularization parameter, by default None

        Returns
        --------
        numpy.floating
            squared residual norm
        """
        return norm((1.0 - self.w(beta=beta)) * self._ub) ** 2.0

    def eta(self, beta: float | None = None) -> np.floating:
        """
        calculate squared regularization norm :math:`\\eta = ||L(x - x_0)||^2`

        Parameters
        ----------
        beta
            regularization parameter, by default ``self._beta``

        Returns
        -------
        numpy.floating
            squared regularization norm
        """
        return norm((self.w(beta=beta) / self._s) * self._ub) ** 2.0

    def eta_diff(self, beta: float | None = None) -> np.floating:
        """calculate differential of `eta` by regularization parameter.

        Parameters
        ----------
        beta
            regularization parameter, by default ``self._beta``

        Returns
        -------
        numpy.floating
            differential of squared regularization norm
        """
        if beta is None:
            beta = self._beta
        w = self.w(beta=beta)
        return (-2.0 / beta) * norm(np.sqrt(1.0 - w) * (w / self._s) * self._ub) ** 2.0

    def residual_norm(self, beta: float | None = None) -> NDArray[float64]:
        """Return the residual norm :math:`\\sqrt{\\rho} = ||Ax - b||`

        Parameters
        ----------
        beta
            reguralization parameter, by default ``self._beta``

        Returns
        -------
        float
            residual norm
        """
        return np.sqrt(self.rho(beta=beta))

    def regularization_norm(self, beta: float | None = None) -> float:
        """Return the residual norm :math:`\\sqrt{\\eta} = ||L (x - x_0)||`

        Parameters
        ----------
        beta
            reguralization parameter, by default ``self._beta``

        Returns
        -------
        float
            regularization norm
        """
        return np.sqrt(self.eta(beta=beta))

    # -------------------------------------------------------------------------
    # calculating the solution using tikhonov - phillips regularization using SVD
    # -------------------------------------------------------------------------

    def inverted_solution(self, beta: float | None = None) -> NDArray[float64]:
        """calculate the inverted solution using given regularization
        parameter.

        Parameters
        ----------
        beta
            regularization parameter, by default None

        Returns
        -------
        numpy.ndarray
            solution vector
        """
        w = self.w(beta=beta)
        if self._inversion_base_vectors is None:
            if self._L_inv is not None:
                self.inversion_base_vectors = self._L_inv @ self.vh.T
            else:
                self.inversion_base_vectors = np.transpose(self.vh)
        return np.dot(self._inversion_base_vectors, (w / self._s) * self._ub)
