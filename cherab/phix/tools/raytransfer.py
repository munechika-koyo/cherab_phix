"""Module to offer a raytransfer objects."""
from __future__ import annotations

import numpy as np
from cherab.tools.equilibrium import EFITEquilibrium
from cherab.tools.raytransfer import RayTransferCylinder
from raysect.core import Node, translate

from cherab.phix.plasma import import_equilibrium

__all__ = ["import_phix_rtc"]


def import_phix_rtc(
    parent: Node, equilibrium: EFITEquilibrium | None = None, grid_size: float = 2.0e-3
) -> RayTransferCylinder:
    """This is a helper function to easily set up the RayTransfer Cylinder
    object on PHiX configuration.

    This function returns a instance of
    :obj:`~cherab.tools.raytransfer.raytransfer.RayTransferCylinder` object,
    the mask property of which is defined to cut out the limiter.
    The spatial resolution of the grid is only in the radial and Z-axis directions.

    Parameters
    ----------
    parent
        Raysect's scene-graph parent node
    equilibrium
        :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium` object, by default
        the instance loaded by `.import_equilibrium` function.
    grid_size
        1 voxel size :math:`(dr=dz)`, by default ``2.0e-3`` [m]

    Return
    ------
    :obj:`~cherab.tools.raytransfer.raytransfer.RayTransferCylinder`
        cherab's Ray Transfer Cylinder object

    Example
    -------
    .. prompt:: python >>> auto

        >>> from raysect.optical import World
        >>> from cherab.phix.tools.raytransfer import import_phix_rtc
        >>> world = World()
        >>> rtc = import_phix_rtc(world)
        >>> rtc.bins
        13326

    The grid shape and steps is as follows:

    .. prompt:: python >>> auto

        >>> rtc.material.grid_shape
        (90, 1, 165)

        >>> rtc.material.grid_steps
        (0.002, 360.0, 0.002)
    """
    # check arguments
    if equilibrium is None:
        equilibrium = import_equilibrium()

    if not isinstance(equilibrium, EFITEquilibrium):
        message = "The equilibrium argument must be a valid EFITEquilibrium object."
        raise TypeError(message)

    # set Ray Transfer Matrix
    r_min = equilibrium.limiter_polygon[:, 0].min()
    z_min = equilibrium.limiter_polygon[:, 1].min()
    d_r = equilibrium.limiter_polygon[:, 0].max() - r_min
    d_z = equilibrium.limiter_polygon[:, 1].max() - z_min

    rtc = RayTransferCylinder(
        radius_outer=r_min + d_r,
        height=d_z,
        n_radius=int(round(d_r / grid_size)),
        n_height=int(round(d_z / grid_size)),
        radius_inner=r_min,
        parent=parent,
        transform=translate(0, 0, z_min),
    )

    # cut out the limiter
    mask = np.zeros(rtc.material.grid_shape)
    for ir, r in enumerate(
        np.arange(
            r_min + 0.5 * rtc.material.dr, r_min + d_r + 0.5 * rtc.material.dr, rtc.material.dr
        )
    ):
        for iz, z in enumerate(
            np.arange(
                z_min + 0.5 * rtc.material.dz, z_min + d_z + 0.5 * rtc.material.dz, rtc.material.dz
            )
        ):
            mask[ir, 0, iz] = equilibrium.inside_limiter(r, z)
    rtc.mask = mask.astype(bool)

    return rtc
