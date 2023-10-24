"""Module to offer the function to generate a derivative matrix."""
import numpy as np
from scipy.sparse import lil_matrix

cimport cython
from numpy cimport import_array, ndarray

__all__ = ["compute_dmat"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef object compute_dmat(
    ndarray voxel_map,
    str kernel_type="laplacian4",
    ndarray kernel=None,
):
    """Generate derivative sparse matrix.

    Parameters
    ----------
    voxel_map : numpy.ndarray
        (N, M) voxel map matrix (negative value must be input into masked voxels)
        If the additional dimension size of the matrix is 1, then it is squeezed to a 2-D matrix.
    kernel_type : {"x", "y", "laplacian4", "laplacian8", "custom"}, optional
        Derivative kernel type. Default is "laplacian8".
        `"custom"` is available only when `.kernel` is specified.
    kernel : numpy.ndarray, optional
        (3, 3) custom kernel matrix. Default is None.

    Returns
    -------
    :obj:`scipy.sparse.csc_matrix`
        (N, N) derivative Compressed Sparse Column matrix (if N > M)

    Notes
    -----
    The derivative matrix is generated by the kernel convolution method.
    The kernel is a 3x3 matrix, and the convolution is performed as follows:

    .. math::

        I_{x, y}' = \\sum_{i=-1}^{1}\\sum_{j=-1}^{1} K_{i,j} \\times I_{x + i, y + j},

    where :math:`I_{x, y}` is the 2-D image at the point :math:`(x, y)` and :math:`K_{i,j}` is the
    kernel matrix.
    Using derivative kernel like a laplacian filter, the derivative matrix defined as follows is
    generated:

    .. math::

        \\mathbf{I}' = \\mathbf{K} \\cdot \\mathbf{I},

    where :math:`\\mathbf{I}` is the vecotrized image and :math:`\\mathbf{K}` is the derivative
    matrix.

    The implemented derivative kernels are as follows:

    - First derivative in x-direction (`kernel_type="x"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 0 & 0 & 0 \\\\ -1 & 1 & 0 \\\\ 0 & 0 & 0 \\end{bmatrix}`
    - First derivative in y-direction (`kernel_type="y"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 0 & -1 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 0 \\end{bmatrix}`
    - Laplacian-4 (`kernel_type="laplacian4"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}`
    - Laplacian-8 (`kernel_type="laplacian8"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & -8 & 1 \\\\ 1 & 1 & 1 \\end{bmatrix}`


    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.optical import World
        >>> from cherab.phix.tools.raytransfer import import_phix_rtc
        >>> from cherab.phix.tools import compute_dmat
        >>>
        >>> world = World()
        >>> rtc = import_phix_rtc(world)
        >>>
        >>> laplacian = compute_dmat(rtc.voxel_map)
        >>> laplacian.toarray()
        array([[-8.,  1.,  0., ...,  0.,  0.,  0.],
               [ 1., -8.,  1., ...,  0.,  0.,  0.],
               [ 0.,  1., -8., ...,  0.,  0.,  0.],
               ...,
               [ 0.,  0.,  0., ..., -8.,  1.,  0.],
               [ 0.,  0.,  0., ...,  1., -8.,  1.],
               [ 0.,  0.,  0., ...,  0.,  1., -8.]])
    """
    cdef:
        int i, j, x, y, row, col, map_mat_max
        double[3][3] kernel_carray
        ndarray map_matrix
        object dmatrix
        double[:, ::] kernel_mv
        int[:, ::] map_matrix_mv

    # define derivative kernel
    if kernel_type == "x":
        kernel_carray[0][:] = [0, 0, 0]
        kernel_carray[1][:] = [-1, 1, 0]
        kernel_carray[2][:] = [0, 0, 0]
        kernel_mv = kernel_carray

    elif kernel_type == "y":
        kernel_carray[0][:] = [0, -1, 0]
        kernel_carray[1][:] = [0, 1, 0]
        kernel_carray[2][:] = [0, 0, 0]
        kernel_mv = kernel_carray

    elif kernel_type == "laplacian4":
        kernel_carray[0][:] = [0, 1, 0]
        kernel_carray[1][:] = [1, -4, 1]
        kernel_carray[2][:] = [0, 1, 0]
        kernel_mv = kernel_carray

    elif kernel_type == "laplacian8":
        kernel_carray[0][:] = [1, 1, 1]
        kernel_carray[1][:] = [1, -8, 1]
        kernel_carray[2][:] = [1, 1, 1]
        kernel_mv = kernel_carray

    elif kernel_type == "custom":
        if kernel is None:
            raise ValueError("kernel must be specified when kernel_type is 'custom'")
        else:
            if kernel.ndim != 2:
                raise ValueError("kernel must be 2-D matrix")

            elif kernel.shape[0] != 3 or kernel.shape[1] != 3:
                raise ValueError("kernel must be 3x3 matrix")

            else:
                kernel_mv = kernel.astype(float)

    else:
        raise ValueError("kernel must be 'x', 'y', 'laplacian4', 'laplacian8' or 'custom'")

    # padding voxel map boundary by -1
    voxel_map = np.squeeze(voxel_map)
    if voxel_map.ndim == 2:
        map_matrix = np.pad(np.squeeze(voxel_map), pad_width=1, constant_values=-1)
        map_mat_max = map_matrix.max()
    else:
        raise ValueError("voxel_map must be 2-D matrix")

    # define derivative matrix as a sparse matrix
    dmatrix = lil_matrix((map_mat_max + 1, map_mat_max + 1), dtype=float)

    # define memoryview
    map_matrix_mv = map_matrix.astype(np.intc)

    # generate derivative matrix
    for row in range(map_mat_max + 1):
        (x,), (y,) = np.where(map_matrix == row)  # TODO: replace to cythonic codes
        for i in range(-1, 1 + 1):
            for j in range(-1, 1 + 1):
                col = map_matrix_mv[x + i, y + j]
                if col > -1:
                    dmatrix[row, col] = kernel_mv[i + 1, j + 1]
                else:
                    pass

    return dmatrix.tocsc()
