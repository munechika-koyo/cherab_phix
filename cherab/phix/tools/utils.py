# definition of useful miscellaneous functions

import numpy as np


def profile_1D_to_2D(data_1D, rtm=None):
    """convert 1D vector to 2D array according to raytrasfer object voxel_map.

    Parameters
    ----------
    data_1D : 1D-array (vector-like)
        1D array data
    rtm : cherab's RayTrasfer Object, required
        cherab's raytransfer object

    Returns
    -------
    data_2D : numpy.ndarray
        2D array converted from 1D vector
    """
    data_2D = np.zeros(rtm.material.grid_shape[::2])
    for i in range(rtm.bins):
        index = np.where(rtm.voxel_map.squeeze() == i)
        data_2D[index] = data_1D[i]

    return data_2D


def profile_2D_to_1D(data_2D, rtm=None):
    """convert 2D array to 1D vector according to raytrasfer object voxel_map.

    Parameters
    ----------
    data_2D : 2Darray (vector-like)
        1D array data
    rtm : cherab's RayTrasfer Object, required
        cherab's raytransfer object

    Returns
    -------
    data_1D : numpy.ndarray
        (N, ) 1D array converted from data_2D
    """
    map_matrix = np.squeeze(rtm.voxel_map)
    data_1D = np.zeros(rtm.bins)
    for i in range(rtm.bins):
        index = np.where(map_matrix == i)  # transform voxel 2D index into 1D rtm.bins index
        data_1D[i] = data_2D[index][0]

    return data_1D
