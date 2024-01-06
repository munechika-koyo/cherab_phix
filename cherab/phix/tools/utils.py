"""Module for useful miscellaneous functions."""
from __future__ import annotations

import numpy as np
from contourpy import contour_generator
from numpy.typing import NDArray
from raysect.optical import World

from cherab.tools.raytransfer import RayTransferCylinder

from .raytransfer import import_phix_rtc

__all__ = ["profile_1D_to_2D", "profile_2D_to_1D", "calc_contours"]


def profile_1D_to_2D(data_1D: NDArray, rtc: RayTransferCylinder) -> NDArray:
    """Convert 1D vector to 2D array according to raytrasfer object's ``invert_voxel_map``.

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
    if data_1D.ndim != 1:
        raise ValueError("data_1D must be 1D array")

    indices = rtc.invert_voxel_map()
    data_array = np.zeros(rtc.material.grid_shape)
    for index, data in zip(indices, data_1D):  # noqa: B905
        data_array[index] = data

    return data_array.squeeze()


def profile_2D_to_1D(
    data_2D: NDArray,
    rtc: RayTransferCylinder,
) -> NDArray:
    """Convert 2D array to 1D vector according to raytrasfer object's ``voxel_map``.

    Parameters
    ----------
    data_2D : numpy.ndarray (N, M)
        2D array data, the shape of which must be same as a 2-D voxel map
    rtc
        cherab's raytransfer cylinder object

    Returns
    -------
    numpy.ndarray
        (N, ) 1D array converted from data_2D
    """
    if data_2D.ndim != 2:
        raise ValueError("data_2D must be 2D array")

    voxel_map = rtc.voxel_map.squeeze()
    if voxel_map.shape != data_2D.shape:
        raise ValueError("data_2D shape must be same as raytransfer's voxel_map")

    data_1D = np.zeros(rtc.bins)
    for row in range(voxel_map.shape[0]):
        for col in range(voxel_map.shape[1]):
            index = voxel_map[row, col]
            if index != -1:
                data_1D[index] = data_2D[row, col]

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

    Returns
    -------
    list[NDArray]
        list of contour lines 2-D array :math:`(N, 2)`.
        The columns of each array show the values of the :math:`R` and :math:`Z` coordinates,
        respectively.
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
