# package the tools for various inversion methods
import numpy as np
from numpy.linalg import norm


class InversionMethod:
    """Inversion calculation based on singular value decomposition (eco algorithum i.e. not full-matrices)
    This provides the users useful tools for regularization computation using SVD components.
    The SVD components are based on the following equation:

    .. math::
        u * s * vh = AL^{-1}

    Parameters
    -----------
    s : vector_like
        singular values in :math:`s`
    u : array_like
        SVD left singular vectors forms as one matrix like :math:`u = (u_1, u_2, ...)`
    vh : array_like
        SVD right singular vectors forms as one matrix like :math:`vh = (v_1, v_2, ...)^T`
    inversion_base_vectors : array-like
        The components of inversions base represented as `L_inv @ vh.T`.
        If this valuable is None, it is automatically computed when computing the inverted solution.
    L_inv : array-like
        inversion matrix in the regularization term. `L_inv` == :math:`L^{-1}` in :math:`||L(x - x_0)||^2`
    data : vector_like
        given data for inversion calculation
    beta : float, optional
        regularization parameter, by default 1.0e-2

    Attributes
    ----------
    L_inv : array-like
        inversion matrix in the regularization term. `L_inv` == :math:`L^{-1}` in :math:`||L(x - x_0)||^2`
        by, default `np.identity(self._vh.shape[1])`
    data : vector_like
        given data for inversion calculation
    inversion_base_vectors : array-like
        The components of inversions base represented as `L_inv @ vh.T`.
    beta : float
        regularization parameter, by default 1.0e-2
    lambdas : 1D array-like
        regularization paramters list, by default `10 ** np.linspace(-5, 5, 100)`

    Methods
    -------
    w : window function
    rho : squared residual norm :math:`||Ax_\\lambda - b||^2`
    eta : squared regularization norm :math:`||Lx_\\lambda||^2`
    eta_dif : the differencial of eta
    residual_norm
    regularization_norm
    inveted_solution : calculate the inverted solution
    """

    def __init__(
        self, s=None, u=None, vh=None, inversion_base_vectors=None, L_inv=None, data=None, beta=None
    ):
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
        if data is not None:
            self.data = data

        # set SVD regularization parameters
        self._lambda = None
        self.beta = beta or 1.0e-2
        self._lambdas = 10 ** np.linspace(-5, 5, 100)

    @property
    def s(self):
        return self._s

    @property
    def u(self):
        return self._u

    @property
    def vh(self):
        return self._vh

    @property
    def L_inv(self):
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
        return self._inversion_base_vectors

    @inversion_base_vectors.setter
    def inversion_base_vectors(self, mat):
        mat = np.asarray(mat)
        if mat.shape[1] != self._s.size:
            raise ValueError(
                "the number of columns of Image Base matrix must be same as the one of singular values"
            )
        self._inversion_base_vectors = mat

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        data = np.array(value, dtype=np.float).ravel()
        if data.size != self._u.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data
        self._ub = np.dot(np.transpose(self._u), self._data)

    @property
    def beta(self):
        return self._lambda

    @beta.setter
    def beta(self, value):
        if not isinstance(value, float):
            raise ValueError("regularization parameter beta must be one float number.")
        self._lambda = value

    @property
    def lambdas(self):
        return self._lambdas

    @lambdas.setter
    def lambdas(self, array):
        if not isinstance(array, np.ndarray) and not isinstance(array, list):
            raise ValueError("lambdas must be the 1D-array of regularization parameters")
        self._lambdas = np.asarray_chkfinite(array)

    # -------------------------------------------------------------------------
    # Define methods calculating some norms, window function, etc...
    # -------------------------------------------------------------------------

    def w(self, beta=None):
        """calculate window function using regularization parameter as a valuable
        and using singular values.

        Parameter
        ---------
        beta : float, optional
            regularization parameter, by default None

        Return
        ------
        numpy.ndarray (N, )
            window function :math:`\\frac{1}{1 + \\lambda / \\sigma_i^2}`
        """
        self._lambda = beta or self._lambda
        return 1.0 / (1.0 + self._lambda / self._s ** 2.0)

    def rho(self, beta=None):
        """calculate squared residual norm :math:`\\rho = ||Ax - b||^2`.

        Parameter
        ---------
        beta : float, optional
            regularization parameter, by default None

        Return
        ------
        numpy.ndarray (N, )
            squared residual norm
        """
        self._lambda = beta or self._lambda
        return norm((1.0 - self.w()) * self._ub) ** 2.0

    def eta(self, beta=None):
        """calculate squared regularization norm :math:`\\eta = ||L(x - x_0)||^2`

        Parameter
        ---------
        beta : float, optional
            regularization parameter, by default self._lambda

        Return
        ------
        numpy.ndarray (N, )
            squared regularization norm
        """
        self._lambda = beta or self._lambda
        return norm((self.w() / self._s) * self._ub) ** 2.0

    def eta_diff(self, beta=None):
        """calculate differential of `eta` by regularization parameter

        Parameter
        ---------
        beta : float, optional
            regularization parameter, by default self._lambda

        Return
        ------
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
            reguralization parameter, by default self._lambda

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
            reguralization parameter, by default self._lambda

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

        Return
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
