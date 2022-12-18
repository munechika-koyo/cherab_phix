"""Module for useful miscellaneous functions."""
from __future__ import annotations

import numpy as np
from cherab.tools.raytransfer import RayTransferCylinder
from contourpy import contour_generator
from numpy.typing import NDArray
from raysect.optical import World

from cherab.phix.tools.raytransfer import import_phix_rtc

__all__ = ["profile_1D_to_2D", "profile_2D_to_1D", "calc_contours"]


def profile_1D_to_2D(data_1D, rtc: RayTransferCylinder) -> NDArray:
    """convert 1D vector to 2D array according to raytrasfer object's
    ``voxel_map``.

    Parameters
    ----------
    data_1D : 1D-array (vector-like)
        1D array data
    rtc
        cherab's raytransfer cylinder object

    Returns
    -------
    numpy.ndarray
        2D array converted from 1D vector
    """
    data_2D = np.zeros(rtc.material.grid_shape[::2])
    for i in range(rtc.bins):
        index = np.where(rtc.voxel_map.squeeze() == i)
        data_2D[index] = data_1D[i]

    return data_2D


def profile_2D_to_1D(
    data_2D,
    rtc: RayTransferCylinder,
) -> NDArray:
    """convert 2D array to 1D vector according to raytrasfer object's
    ``voxel_map``.

    Parameters
    ----------
    data_2D : 2Darray (vector-like)
        1D array data
    rtc
        cherab's raytransfer cylinder object

    Returns
    -------
    numpy.ndarray
        (N, ) 1D array converted from data_2D
    """
    map_matrix = np.squeeze(rtc.voxel_map)
    data_1D = np.zeros(rtc.bins)
    for i in range(rtc.bins):
        index = np.where(map_matrix == i)  # transform voxel 2D index into 1D rtc.bins index
        data_1D[i] = data_2D[index][0]

    return data_1D


def calc_contours(
    profile: NDArray,
    level: float,
    r: NDArray | None = None,
    z: NDArray | None = None,
    rtc: RayTransferCylinder | None = None,
) -> list[NDArray]:
    """Calculate :math:`R-Z` 2-D profile's contours using :obj:`contourpy`.

    Parameters
    ----------
    profile
        :math:`R-Z` 2-D array. The shape of array is :math:`(N_R, N_Z)`.
    level
        contour level of a profile value
    r
        The r-coorinate of the profile values. The default is calculated by `rtc`.
    z
        The z-coorinate of the profile values. The default is calculated by `rtc`.
    rtc
        RayTransferCylinder instance, by default returned instance of `.import_phix_rtc`.

    Return
    ------
    list[NDArray]
        list of contour lines 2-D array
    """
    # calculate r, z vertices
    if z is None or r is None:
        if rtc is None:
            world = World()
            rtc = import_phix_rtc(world)

        z = np.linspace(-1 * rtc.transform[2, 3], rtc.transform[2, 3], rtc.material.grid_shape[2])
        r = np.linspace(
            rtc.material.rmin,
            rtc.material.rmin + rtc.material.dr * rtc.material.grid_shape[0],
            rtc.material.grid_shape[0],
        )

    # create contour generator
    cont_gen = contour_generator(x=r, y=z, z=np.flipud(profile.T))

    return cont_gen.lines(level)
