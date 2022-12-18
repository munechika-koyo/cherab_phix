"""Module to offer the function to generate a laplacian matrix."""
import numpy as np

cimport cython
from numpy cimport import_array, int32_t, ndarray

__all__ = ["laplacian_matrix"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef ndarray[int32_t, ndim=2] laplacian_matrix(
    ndarray voxel_map,
    int dir=4,
):
    """generate Laplacian matrix.

    Parameters
    ----------
    voxel_map : NDArray[int32]
        (N, M) voxel map matrix (negative value must be input into masked voxels)
        If the additional dimension size of the matrix is 1, then it is squeezed to a 2-D matrix.
    dir : int, optional
        the number of laplacian kernel's neighbor in {4, 8}, by default 4.

    Return
    ------
    NDArray[int32]
        (N, N) laplacian matrix (if N > M)

    Example
    -------
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
        >>> laplacian
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
        ndarray[int32_t, ndim=2] map_matrix
        ndarray[int32_t, ndim=2] laplacian_mat
        int[:, ::] map_matrix_mv
        int[:, ::] laplacian_mat_mv

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
    map_mat_max = np.max(map_matrix)

    # define laplacian matrix
    laplacian_mat = np.zeros((map_mat_max + 1, map_mat_max + 1), dtype=np.int32)

    # define memoryview
    map_matrix_mv = map_matrix
    laplacian_mat_mv = laplacian_mat

    # generate laplacian matrix
    for row in range(map_mat_max + 1):
        x, y = np.where(map_matrix == row)  # TODO: replace to cythonic codes
        for i in range(-1, 1 + 1):
            for j in range(-1, 1 + 1):
                col = map_matrix_mv[x + i, y + j]
                if col > -1:
                    laplacian_mat_mv[row, col] = kernel[i + 1][j + 1]
                else:
                    pass

    return laplacian_mat
