# package the tools for various inversion methods
import numpy as np
from numpy.linalg import norm


class SVDInversionBase:
    """Base class for inversion calculation based on singular value decomposition (eco algorithum i.e. not full-matrices).
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

        u * s * vh = AL^{-1}

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
    beta : float, optional
        regularization parameter, by default 1.0e-2
    """

    def __init__(self, s, u, vh, data, inversion_base_vectors=None, L_inv=None, beta=1.0e-2):
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

        # initial data values and dot(u, b)
        self._data = None
        self._ub = None
        self.data = data

        # set SVD regularization parameters
        self._lambda = None
        self.beta = beta

    @property
    def s(self):
        """
        vector_like: singular values :math:`\\sigma_i` in :math:`s` vectors.
        """
        return self._s

    @property
    def u(self):
        """
        array_like: SVD left singular vectors forms as one matrix like :math:`u = (u_1, u_2, ...)`
        """
        return self._u

    @property
    def vh(self):
        """
        array_like: SVD right singular vectors forms as one matrix like :math:`vh = (v_1, v_2, ...)^T`
        """
        return self._vh

    @property
    def L_inv(self):
        """
        :obj:`numpy.ndarray`: inversion matrix in the regularization term. :obj:`L_inv` is :math:`L^{-1}` in :math:`||L(x - x_0)||^2`,
        by default ``numpy.identity(self._vh.shape[1])``
        """
        return self._L_inv

    @L_inv.setter
    def L_inv(self, value):
        inv_mat = np.asarray(value)
        n, m = inv_mat.shape
        if m != self._vh.shape[1] & n == m:
            raise ValueError("L_inv must be a square ")
        self._L_inv = inv_mat

    @property
    def inversion_base_vectors(self):
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
            raise ValueError("the number of columns of Image Base matrix must be same as the one of singular values")
        self._inversion_base_vectors = mat

    @property
    def data(self):
        """
        :obj:`numpy.ndarray`: given data for inversion calculation
        """
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray_chkfinite(value, dtype=np.float).ravel()
        if data.size != self._u.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data
        self._ub = np.dot(np.transpose(self._u), self._data)

    @property
    def beta(self):
        """
        float: regularization parameter
        """
        return self._lambda

    @beta.setter
    def beta(self, value):
        if not isinstance(value, float):
            raise ValueError("regularization parameter beta must be one float number.")
        self._lambda = value

    # -------------------------------------------------------------------------
    # Define methods calculating some norms, window function, etc...
    # -------------------------------------------------------------------------

    def w(self, beta=None):
        """calculate window function using regularization parameter as a valuable
        and using singular values.

        Parameters
        -----------
        beta : float, optional
            regularization parameter, by default None

        Returns
        --------
        numpy.ndarray (N, )
            window function :math:`\\frac{1}{1 + \\lambda / \\sigma_i^2}`
        """
        self._lambda = beta or self._lambda
        return 1.0 / (1.0 + self._lambda / self._s ** 2.0)

    def rho(self, beta=None):
        """
        calculate squared residual norm :math:`\\rho = ||Ax - b||^2`.

        Parameters
        ----------
        beta : float, optional
            regularization parameter, by default None

        Returns
        --------
        numpy.ndarray (N, )
            squared residual norm
        """
        self._lambda = beta or self._lambda
        return norm((1.0 - self.w()) * self._ub) ** 2.0

    def eta(self, beta=None):
        """
        calculate squared regularization norm :math:`\\eta = ||L(x - x_0)||^2`

        Parameters
        ----------
        beta : float, optional
            regularization parameter, by default ``self._lambda``

        Returns
        -------
        numpy.ndarray (N, )
            squared regularization norm
        """
        self._lambda = beta or self._lambda
        return norm((self.w() / self._s) * self._ub) ** 2.0

    def eta_diff(self, beta=None):
        """calculate differential of `eta` by regularization parameter

        Parameters
        ----------
        beta : float, optional
            regularization parameter, by default ``self._lambda``

        Returns
        -------
        numpy.ndarray (N, )
            differential of squared regularization norm
        """
        self._lambda = beta or self._lambda
        w = self.w()
        return (-2.0 / self._lambda) * norm(np.sqrt(1.0 - w) * (w / self._s) * self._ub) ** 2.0

    def residual_norm(self, beta=None):
        """Return the residual norm :math:`\\sqrt{\\rho} = ||Ax - b||`

        Parameters
        ----------
        beta : float, optional
            reguralization parameter, by default ``self._lambda``

        Returns
        -------
        numpy.ndarray (N, )
            residual norm
        """
        return np.sqrt(self.rho(beta))

    def regularization_norm(self, beta=None):
        """Return the residual norm :math:`\\sqrt{\\eta} = ||L (x - x_0)||`

        Parameters
        ----------
        beta : float, optional
            reguralization parameter, by default ``self._lambda``

        Returns
        -------
        numpy.ndarray (N, )
            regularization norm
        """
        return np.sqrt(self.eta(beta))

    # -------------------------------------------------------------------------
    # calculating the solution using tikhonov - phillips regularization using SVD
    # -------------------------------------------------------------------------

    def inverted_solution(self, beta=None):
        """calculate the inverted solution using given regularization parameter.

        Parameters
        ----------
        beta : float, optional
            regularization parameter, by default None

        Returns
        -------
        numpy.ndarray
            solution vector
        """
        self._lambda = beta or self._lambda
        w = self.w()
        if self._inversion_base_vectors is None:
            if self._L_inv is not None:
                self.inversion_base_vectors = self._L_inv @ self.vh.T
            else:
                self.inversion_base_vectors = np.transpose(self.vh)
        return np.dot(self._inversion_base_vectors, (w / self._s) * self._ub)
