"""Module to offer the helper function to populate an EFITequilibrium
instance."""
import json
from pathlib import Path

from cherab.tools.equilibrium.efit import EFITEquilibrium
from raysect.core import Point2D

__all__ = ["import_equilibrium"]


def import_equilibrium(model_variant: str = "phix10") -> EFITEquilibrium:
    """Return a populated instance of the PHiX equilibrium calculated by
    Tokamak Simulation Code.

    Parameters
    ------------
    model_variant
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
    example_file = Path(__file__).parent.resolve() / "data" / f"{model_variant}.json"
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
    limiter_polygon = eq_data["limiter_polygon"]
    time = eq_data["time"]

    equilibrium = EFITEquilibrium(
        r,
        z,
        psi,
        psi_axis,
        psi_lcfs,
        axis_coord,
        x_points,
        strike_points,
        f_profile,
        q_profile,
        b_vacuum_radius,
        b_vacuum_magnitude,
        lcfs_polygon,
        limiter_polygon,
        time,
    )

    return equilibrium


# For debugging
if __name__ == "__main__":
    eq = import_equilibrium(model_variant="phix12")
    pass
