"""Inverses provided data using Minimum Fisher Regularization (MFR) scheme."""
from __future__ import annotations

import pickle
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Type

import numpy as np
from scipy.sparse import csc_matrix, issparse, spmatrix
from scipy.sparse import diags as spdiags

from ..tools import Spinner
from .inversion import _SVDBase, compute_svd
from .lcurve import Lcurve

__all__ = ["Mfr"]


class Mfr:
    """Inverses provided data using Minimum Fisher Regularization (MFR) scheme.

    Parameters
    ----------
    gmat : numpy.ndarray (M, N) | scipy.sparse.spmatrix (M, N)
        matrix :math:`T` of the forward problem (geometry matrix, ray transfer matrix, etc.)
    dmats : list[tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]]
        list of pairs of derivative matrices :math:`D_i` and :math:`D_j` along to :math:`i` and
        :math:`j` coordinate directions, respectively
    data : numpy.ndarray (M, ), optional
        given data for inversion calculation, by default None

    Notes
    -----
    Minimum Fisher Regularization (MFR) was firstly introduced by [1]_ for the tomographic
    reconstruction of the emissivity in the TCV tokamaks and later applied to the ASDEX Upgrade
    tokamak [2]_.
    The considered inverse problem is the tomographic reconstruction of the emissivity:

    .. math::

        T\\varepsilon = s,

    where :math:`T\\in\\mathbb{R}^{m\\times n}` is the geometry matrix describing the spatial
    distribution of detectors, :math:`\\varepsilon\\in\\mathbb{R}^n` is the solution emissivity
    vector, and :math:`s\\in\\mathbb{R}^m` is the signal given data vector.

    Since this problem is often ill-posed, one requires a regularization to be solved by adding
    an objective functional
    :math:`O(\\varepsilon) \\equiv \\varepsilon^\\mathsf{T} H(\\varepsilon) \\varepsilon`
    to the least square problem:

    .. math::

        \\varepsilon_\\lambda
            := \\text{argmin}
                \\left\\{ ||T\\varepsilon - s||^2
                    + \\lambda\\cdot \\varepsilon^\\mathsf{T} H(\\varepsilon) \\varepsilon
                \\right\\}

    where :math:`\\lambda` is a regularization parameter and :math:`H(\\varepsilon)` is the
    regularization matrix.

    In the MFR scheme, the regularization matrix is defined as:

    .. math::

        H(\\varepsilon) &:= \\sum_{i,j} \\alpha_{ij} D_i^\\mathsf{T} W(\\varepsilon) G^{ij} D_j

        W(\\varepsilon) &:= \\text{diag}
            \\left(
                \\frac{1}{\\max\\left\\{\\varepsilon_i, \\epsilon_0\\right\\}}
            \\right),

    where  :math:`\\alpha_{ij}` is the anisotropic coefficient, :math:`D_i` and :math:`D_j` are
    derivative matrices along to :math:`i` and :math:`j` coordinate directions, respectively,
    :math:`W` is the diagonal weight matrix defined as the inverse of :math:`\\varepsilon_i`,
    :math:`\\epsilon_0` is a small number to avoid division by zero and to push the solution to
    be positive, and :math:`G^{i, j}` is the metric tensor matrix between :math:`i` and :math:`j`
    coordinate directions which is the identity matrix in the orthogonal coordinate system.

    The MFR scheme is a non-linear equation, so it is solved by the iterative method optimizing
    the regularization parameter :math:`\\lambda` at each iteration.

    References
    ----------
    .. [1] M Anton, H Weisen, M J Dutch, W von der Linden, F Buhlmann, R Chavan, B Marletaz,
        P Marmillod and P Paris, *X-ray tomography on the TCV tokamak*, Plasma Phys. Control.
        Fusion **38** 1849 (1996), :doi:`10.1088/0741-3335/38/11/001`

    .. [2] Odstrčil T, Pütterich T, Odstrčil M, Gude A, Igochine V, Stroth U; ASDEX Upgrade Team,
        *Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade
        tokamak*, Rev. Sci. Instrum. **87**, 123505 (2016), :doi:`10.1063/1.4971367`
    """

    def __init__(
        self, gmat: np.ndarray | spmatrix, dmats: list[tuple[spmatrix, spmatrix]], data=None
    ):
        # validate arguments
        if not isinstance(gmat, (np.ndarray, spmatrix)):
            raise TypeError("gmat must be a numpy array or a scipy sparse matrix")
        if gmat.ndim != 2:
            raise ValueError("gmat must be a 2D array")

        if not isinstance(dmats, list):
            raise TypeError("dmats must be a list of tuples")
        for dmat1, dmat2 in dmats:
            if not issparse(dmat1):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")
            if not issparse(dmat2):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")

        # set matrix attributes
        self._gmat = gmat
        self._dmats = dmats

        # set data attribute
        self.data = data

    @property
    def gmats(self) -> np.ndarray | spmatrix:
        """Geometry matrix :math:`T` of the forward problem."""
        return self._gmat

    @property
    def dmats(self) -> list[tuple[spmatrix, spmatrix]]:
        """List of pairs of derivative matrices :math:`D_i` and :math:`D_j` along to :math:`i` and
        :math:`j` coordinate directions, respectively."""
        return self._dmats

    @property
    def data(self) -> np.ndarray:
        """Given data for inversion calculation."""
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._gmat.shape[0]:
            raise ValueError("data size must be the same as the number of rows of U matrix")
        self._data = data

    def solve(
        self,
        x0: np.ndarray | None = None,
        derivative_weights: list[float] | None = None,
        eps: float = 1.0e-6,
        bounds: tuple[float, float] = (-20.0, 2.0),
        tol: float = 1e-3,
        miter: int = 20,
        regularizer: Type["_SVDBase"] = Lcurve,
        store_regularizers: bool = False,
        path: str | Path | None = None,
        use_gpu: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Solves the inverse problem using MFR scheme.

        Parameters
        ----------
        x0 : numpy.ndarray
            initial solution vector, by default ones vector
        derivative_weights
            allows to specify anisotropy by assign weights for each matrix, by default ones vector
        eps
            small number to avoid division by zero, by default 1e-6
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
        tol
            tolerance for solution convergence, by default 1e-3
        miter
            maximum number of MFR iterations, by default 20
        regularizer
            regularizer class to use, by default :obj:`~cherab.phix.inversion.Lcurve`
        store_regularizers
            if True, store regularizer objects at each iteration, by default False.
            The path to store regularizer objects can be specified using `path` argument.
        path
            directory path to store regularizer objects, by default None.
            If `path` is None, the regularizer objects will be stored in the current directory
            if `store_regularizers` is True.
        use_gpu
            same as :obj:`~cherab.phix.inversion.inversion.compute_svd`'s `use_gpu` argument,
            by default True
        verbose
            If True, print iteration information regarding SVD computation, by default False
        **kwargs
            additional keyword arguments passed to the regularizer class's :meth:`solve` method
        """
        # validate regularizer
        if not issubclass(regularizer, _SVDBase):
            raise TypeError("regularizer must be a subclass of _SVDBase")

        # check initial solution
        if x0 is None:
            x0 = np.ones(self._gmat.shape[1])
        elif isinstance(x0, np.ndarray):
            if x0.ndim != 1:
                raise ValueError("Initial solution must be a 1D array")
            if x0.shape[0] != self._gmat.shape[1]:
                raise ValueError("Initial solution must have same size as the rows of gmat")
        else:
            raise TypeError("Initial solution must be a numpy array")

        # check store_regularizers
        if store_regularizers:
            if path is None:
                path: Path = Path.cwd()
            else:
                path: Path = Path(path)

        # MFR loop
        niter = 0
        status = {}
        self._converged = False
        errors = []
        start_time = time()
        while niter < miter and not self._converged:
            with Spinner(f"{niter:02}-th MFR iteration", timer=True) as sp:
                sp_base_text = sp.text + " "

                # compute regularization matrix
                hmat = self.regularization_matrix(
                    x0, eps=eps, derivative_weights=derivative_weights
                )

                # compute SVD components
                spinner = sp if verbose else None
                singular, u_vecs, basis = compute_svd(self._gmat, hmat, use_gpu=use_gpu, sp=spinner)

                # find optimal solution using regularizer class
                sp.text = sp_base_text + " (Solving regularizer)"
                reg = regularizer(singular, u_vecs, basis, data=self._data)
                x, _ = reg.solve(bounds=bounds, **kwargs)

                # check convergence
                diff = np.linalg.norm(x - x0)
                errors.append(diff)
                self._converged = bool(diff < tol)

                # update solution
                x0 = x

                # store regularizer object at each iteration
                if store_regularizers:
                    with (path / f"regularizer_{niter}.pickle").open("wb") as f:
                        pickle.dump(reg, f)

                # print iteration information
                _text = f"(Diff: {diff:.3e}, Tolerance: {tol:.3e}, lambda: {reg.lambda_opt:.3e})"
                sp.text = sp_base_text + _text
                sp.ok()

                # update iteration counter
                niter += 1

        elapsed_time = time() - start_time

        # set status
        status["elapsed_time"] = elapsed_time
        status["niter"] = niter
        status["errors"] = errors
        status["converged"] = self._converged
        status["regularizer"] = reg

        print(f"Total elapsed time: {timedelta(seconds=elapsed_time)}")

        return x, status

    def regularization_matrix(
        self,
        x: np.ndarray,
        eps: float = 1.0e-6,
        derivative_weights: list[float] | None = None,
    ) -> csc_matrix:
        """Computes nonlinear regularization matrix from provided derivative matrices and a solution
        vector.

        Multiple derivative matrices can be used allowing to combine matrices computed by
        different numerical schemes.

        Each matrix can have different weight coefficients assigned to introduce anisotropy.

        The expression of the regularization matrix :math:`H(\\varepsilon)` is:

        .. math::

            H(\\varepsilon)
                = \\sum_{i,j} \\alpha_{ij} D_i^\\mathsf{T} W D_j

        where :math:`D_i` and :math:`D_j` are derivative matrices along to
        :math:`i` and :math:`j` coordinate directions, respectively, :math:`\\alpha_{ij}` is the
        anisotropic coefficient, and :math:`W` is the diagonal weight matrix defined as
        the inverse of :math:`\\varepsilon_i`:

        .. math::

            W_{ij} = \\frac{\\delta_{ij}}{ \\max\\left\\{\\varepsilon_i, \\epsilon_0\\right\\} },

        where :math:`\\varepsilon_i` is the i-th element of the solution vector
        :math:`\\varepsilon`, and :math:`\\epsilon_0` is a small number to avoid division by zero
        and to push the solution to be positive.

        Parameters
        ----------
        x : numpy.ndarray
            solution vector :math:`\\varepsilon`
        eps
            small number to avoid division by zero, by default 1.0e-6
        derivative_weights
            allows to specify anisotropy by assign weights for each matrix, by default ones vector

        Returns
        -------
        :obj:`scipy.sparse.csc_matrix`
            regularization matrix :math:`H(\\varepsilon)`
        """
        # validate eps
        if eps <= 0:
            raise ValueError("eps must be positive small number")

        # set weighting matrix
        w = np.zeros_like(x)
        w[x > eps] = 1 / x[x > eps]
        w[x <= eps] = 1 / eps
        w = spdiags(w)

        if derivative_weights is None:
            derivative_weights = [1.0] * len(self._dmats)
        elif len(derivative_weights) != len(self._dmats):
            raise ValueError(
                "Number of derivative weight coefficients must be equal to number of derivative matrices"
            )

        regularization = csc_matrix(self._dmats[0][0].shape, dtype=float)

        for (dmat1, dmat2), aniso in zip(self._dmats, derivative_weights):
            regularization += aniso * dmat1.T @ w @ dmat2

        return regularization
