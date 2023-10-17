"""Module to offer the function to generate a laplacian matrix."""
import numpy as np
from scipy.sparse import lil_matrix

cimport cython
from numpy cimport import_array, ndarray

__all__ = ["laplacian_matrix"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef object laplacian_matrix(
    ndarray voxel_map,
    int dir=4,
):
    """Generate Laplacian sparse matrix.

    Parameters
    ----------
    voxel_map : NDArray[int32]
        (N, M) voxel map matrix (negative value must be input into masked voxels)
        If the additional dimension size of the matrix is 1, then it is squeezed to a 2-D matrix.
    dir : {4, 8}
        the number of laplacian kernel's neighbor, by default 4.

    Returns
    -------
    :obj:`~scipy.sparse.csr_matrix`
        (N, N) laplacian Compressed Sparse Row matrix (if N > M)

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.core import World
        >>> from cherab.phix.plasma import import_equilibrium
        >>> from cherab.phix.tools import laplacian_matrix, import_phix_rtm
        >>>
        >>> world = World()
        >>> eq = import_equilibrium()
        >>> rtm = import_phix_rtm(world, equilibrium=eq)
        >>>
        >>> laplacian = laplacian_matrix(rtm.voxel_map, dir=8)
        >>> laplacian.toarray()
        array([[-8,  1,  0, ...,  0,  0,  0],
               [ 1, -8,  1, ...,  0,  0,  0],
               [ 0,  1, -8, ...,  0,  0,  0],
               ...,
               [ 0,  0,  0, ..., -8,  1,  0],
               [ 0,  0,  0, ...,  1, -8,  1],
               [ 0,  0,  0, ...,  0,  1, -8]], dtype=int32)
    """
    cdef:
        int i, j, x, y, row, col, map_mat_max
        int[3][3] kernel
        ndarray map_matrix
        object laplacian_mat
        int[:, ::] map_matrix_mv

    # define laplacian kernel
    if dir == 4:
        kernel[0][:] = [0, 1, 0]
        kernel[1][:] = [1, -4, 1]
        kernel[2][:] = [0, 1, 0]

    elif dir == 8:
        kernel[0][:] = [1, 1, 1]
        kernel[1][:] = [1, -8, 1]
        kernel[2][:] = [1, 1, 1]

    else:
        raise ValueError("kernel dir allows either 4 or 8.")

    # padding voxel map boundary by -1
    map_matrix = np.pad(np.squeeze(voxel_map), pad_width=1, constant_values=-1)
    map_mat_max = map_matrix.max()

    # define laplacian matrix as a sparse matrix
    laplacian_mat = lil_matrix((map_mat_max + 1, map_mat_max + 1), dtype=np.int32)

    # define memoryview
    map_matrix_mv = map_matrix

    # generate laplacian matrix
    for row in range(map_mat_max + 1):
        (x,), (y,) = np.where(map_matrix == row)  # TODO: replace to cythonic codes
        for i in range(-1, 1 + 1):
            for j in range(-1, 1 + 1):
                col = map_matrix_mv[x + i, y + j]
                if col > -1:
                    laplacian_mat[row, col] = kernel[i + 1][j + 1]
                else:
                    pass

    return laplacian_mat.tocsr()
