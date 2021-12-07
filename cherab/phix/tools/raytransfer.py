import numpy as np
from raysect.core.math import translate
from cherab.tools.equilibrium import EFITEquilibrium
from cherab.tools.raytransfer import RayTransferCylinder


def import_phix_rtm(parent, equilibrium=None, grid_size=2.0e-3):
    """This is a helper function to easily set up the ray transfer matrix object
    on PHiX configuration. This RTM is created by cherab's RayTransferCylinder object,
    and tha mask property of which is defined to cut out the limiter.

    Parameters
    ----------
    parent : :obj:`~raysect.core.scenegraph.node.Node`
        Raysect's scene-graph parent node
    equilibrium : :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium`
        EFITEquilibrium object, by default None
    grid_size : float, optional
        1 voxel size :math:`(dr=dz)`, by default 2.0e-3 [m]

    Returns
    -------
    :obj:`~cherab.tools.raytransfer.raytransfer.RayTransferCylinder`
        cherab's Ray Transfer Cylinder object
    """

    # check arguments
    if not isinstance(equilibrium, EFITEquilibrium):
        message = "The equilibrium argument must be a valid EFITEquilibrium object."
        raise TypeError(message)

    # set Ray Transfer Matrix
    r_min = equilibrium.limiter_polygon[:, 0].min()
    z_min = equilibrium.limiter_polygon[:, 1].min()
    d_r = equilibrium.limiter_polygon[:, 0].max() - r_min
    d_z = equilibrium.limiter_polygon[:, 1].max() - z_min

    rtm = RayTransferCylinder(
        radius_outer=r_min + d_r,
        height=d_z,
        n_radius=int(round(d_r / grid_size)),
        n_height=int(round(d_z / grid_size)),
        radius_inner=r_min,
        parent=parent,
        transform=translate(0, 0, z_min),
    )

    # cut out the limiter
    mask = np.zeros(rtm.material.grid_shape)
    for ir, r in enumerate(
        np.arange(
            r_min + 0.5 * rtm.material.dr, r_min + d_r + 0.5 * rtm.material.dr, rtm.material.dr
        )
    ):
        for iz, z in enumerate(
            np.arange(
                z_min + 0.5 * rtm.material.dz, z_min + d_z + 0.5 * rtm.material.dz, rtm.material.dz
            )
        ):
            mask[ir, 0, iz] = equilibrium.inside_limiter(r, z)
    rtm.mask = mask.astype(np.bool)

    return rtm
