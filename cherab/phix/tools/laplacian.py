import numpy as np


def laplacian_matrix(voxel_map, dir=4, kernel=None):
    """generate Laplacian matrix

    Parameters
    ----------
    voxel_map : (N, M) numpy.ndarray
        voxel map matrix (negative value must be input into masked voxels)
    dir : int,  optional
        the number of laplacian kernel's neighbor , by default 4
        if you want to use another kernel, input it into kernel key word argument.
    kernel : (4, 4) numpy.ndarry, optional
        user-defined kernel, which is supersede

    Return
    --------
    laplacian matrix : (N, N) numpy.ndarray (if N > M)
    """

    # define laplacian kernel
    if dir == 4:
        kernel = kernel or np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif dir == 8:
        kernel = kernel or np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # padding voxel map boundary by -1
    map_matrix = np.pad(voxel_map.squeeze(), pad_width=1, constant_values=-1)

    # define laplacian matrix
    laplacian_mat = np.zeros((map_matrix.max() + 1, map_matrix.max() + 1))

    # generate laplacian matrix
    # TODO: optimize for loop calculation more faster

    for row in range(map_matrix.max() + 1):
        x, y = np.where(map_matrix == row)
        for i in range(-1, 1 + 1):
            for j in range(-1, 1 + 1):
                col = map_matrix[x + i, y + j]
                if col > -1:
                    laplacian_mat[row, col] = kernel[i + 1, j + 1]
                else:
                    pass

    return laplacian_mat
