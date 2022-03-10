import os.path as path
import json

from raysect.core import Point2D
from cherab.tools.equilibrium.efit import EFITEquilibrium
from cherab.phix.machine.wall_outline import INNER_LIMITER


def import_equilibrium(model_variant="phix10"):
    """
    Return a populated instance of the PHiX equilibrium calculated by Tokamak Simulation Code.

    Parameters
    ------------
    model_variant : str
        Name of the equilibrium model variant to load, by default "phix10".
        each data is stored as a .json file in data directory.

    Returns
    -------
    :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium`
        :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium` instance

    Example
    -------
    .. prompt:: python >>> auto

        >>> from cherab.phix.plasma import impot_equilibrium
        >>> equilibrium = impot_equilibrium()
    """
    example_file = path.join(path.dirname(__file__), "data", model_variant + ".json")
    with open(example_file, "r") as fh:
        eq_data = json.load(fh)

    r = eq_data["r"]
    z = eq_data["z"]
    psi = eq_data["psi"]
    psi_axis = eq_data["psi_axis"]
    psi_lcfs = eq_data["psi_lcfs"]
    ac = eq_data["axis_coord"]
    axis_coord = Point2D(*ac)
    xp = eq_data["x_points"]
    x_points = [Point2D(*xp)]
    sp = eq_data["strike_points"]
    strike_points = [Point2D(*sp[0]), Point2D(*sp[1])]
    f_profile = eq_data["f_profile"]
    q_profile = eq_data["q_profile"]
    b_vacuum_radius = eq_data["b_vacuum_radius"]
    b_vacuum_magnitude = eq_data["b_vacuum_magnitude"]
    lcfs_polygon = eq_data["lcfs_polygon"]
    limiter_polygon = INNER_LIMITER[0:-1, :].transpose()
    time = eq_data["time"]

    equilibrium = EFITEquilibrium(
        r, z, psi, psi_axis, psi_lcfs, axis_coord, x_points, strike_points,
        f_profile, q_profile, b_vacuum_radius, b_vacuum_magnitude,
        lcfs_polygon, limiter_polygon, time
    )

    return equilibrium


# For debugging
if __name__ == "__main__":
    eq = import_equilibrium(model_variant="phix12")
    pass
