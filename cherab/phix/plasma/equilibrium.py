"""Module to offer the helper function to populate an EFITequilibrium instance."""
import json
from importlib.resources import files

from raysect.core import Point2D

from cherab.tools.equilibrium.efit import EFITEquilibrium

__all__ = ["load_equilibrium"]


def load_equilibrium(model_variant: str = "phix10") -> EFITEquilibrium:
    """Return a populated instance of the PHiX equilibrium calculated by Tokamak Simulation Code.

    Parameters
    ----------
    model_variant
        Name of the equilibrium model variant to load, by default ``"phix10"``.
        each data is stored as a .json file in data directory.

    Returns
    -------
    :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium`
        :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium` instance

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from cherab.phix.plasma import load_equilibrium
        >>> equilibrium = load_equilibrium()
    """
    model_variants = [f.stem for f in files("cherab.phix.plasma.data").glob("*.json")]  # type: ignore
    model_variants.sort()

    try:
        with files("cherab.phix.plasma.data").joinpath(f"{model_variant}.json").open("r") as f:
            eq_data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"equilibrium data for '{model_variant}' is not found. "
            + f"Please select one from {model_variants}"
        ) from e

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
