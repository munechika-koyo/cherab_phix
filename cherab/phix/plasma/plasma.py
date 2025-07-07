"""Module to offer an helper function to load a plasma object."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from raysect.core import Node, Point3D, Vector3D, translate
from raysect.optical import Spectrum, World
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator
from raysect.primitive import Cylinder, Subtract

from cherab.core import Line, Plasma
from cherab.core.atomic.elements import hydrogen
from cherab.core.distribution import DistributionFunction
from cherab.core.math import VectorAxisymmetricMapper
from cherab.core.model import Bremsstrahlung, ExcitationLine, RecombinationLine
from cherab.core.species import Species
from cherab.openadas import OpenADAS
from cherab.tools.equilibrium import EFITEquilibrium

from ..machine.wall_outline import VESSEL_WALL
from ..tools import Spinner
from .equilibrium import load_equilibrium
from .species import PHiXSpecies

__all__ = ["load_plasma", "emission_hydrogen_balmer"]


def load_plasma(
    parent: Node, eq_model: str = "phix10", species: object | None = None
) -> tuple[Plasma, EFITEquilibrium]:
    """Helper function of generating PHiX plasma.

    The plasma model is constructed by the plasma shape from the equilibrium data, the particle
    species data including each particle's density and temperature profile which was estimated from
    the experiment data, and the emission model of bremsstrahlung and line emission (Hα, Hβ, Hγ, and
    Hδ).

    The equilibrium data is loaded by :func:`.load_equilibrium`.

    Parameters
    ----------
    parent
        Raysect's scene-graph parent node
    eq_model
        equilibrium model name, by default ``"phix10"``.
        This name corresponds to the json file name in the data directory.
    species
        user-defined species object having composition which is a list of
        :obj:`~cherab.core.Species` objects and electron distribution function attributes,
        by default :obj:`.PHiXSpecies`

    Returns
    -------
    tuple[:obj:`~cherab.core.plasma.node.Plasma`, :obj:`~cherab.tools.equilibrium.efit.EFITEquilibrium`]

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.optical import World
        >>> from cherab.phix.plasma import load_plasma
        >>>
        >>> world = World()
        >>> plasma, eq = load_plasma(world)
        ✅ loading plasma ... (data from: phix10)
    """
    with Spinner(f"loading plasma ... (data from: {eq_model})") as sp:
        try:
            # create equilibrium instance
            eq = load_equilibrium(model_variant=eq_model)

            # create atomic data source
            adas = OpenADAS(permit_extrapolation=True)

            # generate plasma object instance
            plasma = Plasma(parent=parent, name="PHiX_plasma")

            # setting plasma properties
            plasma.atomic_data = adas
            plasma.integrator = NumericalIntegrator(step=0.001)
            plasma.b_field = VectorAxisymmetricMapper(eq.b_field)

            # create plasma geometry as subtraction of two cylinders
            inner_radius = VESSEL_WALL[:, 0].min()
            outer_radius = VESSEL_WALL[:, 0].max()
            height = VESSEL_WALL[:, 1].max() - VESSEL_WALL[:, 1].min()

            inner_cylinder = Cylinder(inner_radius, height)
            outer_cylinder = Cylinder(outer_radius, height)

            plasma.geometry = Subtract(outer_cylinder, inner_cylinder)
            plasma.geometry_transform = translate(0, 0, VESSEL_WALL[:, 1].min())

            # apply species to plasma
            if not (hasattr(species, "composition") and hasattr(species, "electron_distribution")):
                species = PHiXSpecies(equilibrium=eq)

            if isinstance(composition := getattr(species, "composition", None), Iterable):
                for element in composition:
                    if not isinstance(element, Species):
                        raise TypeError(
                            "element of composition attr must be a cherab.core.Species object."
                        )
                plasma.composition = composition
            else:
                raise TypeError("composition attr must be an iterable object.")

            if isinstance(
                electron_distribution := getattr(species, "electron_distribution", None),
                DistributionFunction,
            ):
                plasma.electron_distribution = electron_distribution
            else:
                raise TypeError(
                    "electron_distribution must be a cherab.core.DistributionFunction object."
                )

            # apply emission from plasma
            h_alpha = Line(hydrogen, 0, (3, 2))  # , wavelength=656.279)
            h_beta = Line(hydrogen, 0, (4, 2))  # , wavelength=486.135)
            h_gamma = Line(hydrogen, 0, (5, 2))  # , wavelength=434.0472)
            h_delta = Line(hydrogen, 0, (6, 2))  # , wavelength=410.1734)
            # ciii_777 = Line(
            #     carbon, 2, ("1s2 2p(2P°) 3d 1D°", " 1s2 2p(2P°) 3p  1P")
            # )  # , wavelength=770.743)
            plasma.models = [
                Bremsstrahlung(),
                ExcitationLine(h_alpha),
                ExcitationLine(h_beta),
                ExcitationLine(h_gamma),
                ExcitationLine(h_delta),
                # ExcitationLine(ciii_777),
                RecombinationLine(h_alpha),
                RecombinationLine(h_beta),
                RecombinationLine(h_gamma),
                RecombinationLine(h_delta),
                # RecombinationLine(ciii_777),
            ]

            sp.ok()

        except Exception as e:
            sp.fail()
            raise e

    return (plasma, eq)


def emission_hydrogen_balmer(
    r_grid,
    z_grid,
    balmer: str = "alpha",
    spectrum_step: float = 0.01,
    eq_model: str = "phix10",
) -> np.ndarray:
    """Calculate the emissivity of hydrogen Balmer lines.

    Calculate the emissivity of hydrogen Balmer lines (Hα, Hβ, Hγ, and Hδ) at each points of the
    given :math:`R, Z` grid.
    The unit of the emissivity is [W/m^3].
    The plasma object is generated by :func:`.load_plasma`.

    Parameters
    ----------
    r_grid : 1-D array_like
        1-D array of :math:`R` grid points
    z_grid : 1-D array_like
        1-D array of :math:`Z` grid points
    balmer
        Balmer line name, by default ``"alpha"``.
        This name must be one of ``"alpha"``, ``"beta"``, ``"gamma"``, and ``"delta"``.
    spectrum_step
        wavelength step of the spectrum, by default ``0.01`` [nm]
    eq_model
        equilibrium model name, by default ``"phix10"``.
        This parameter corresponds to `.load_plasma`'s ``eq_model`` parameter.

    Returns
    -------
    np.ndarray
        2-D array of the emissivity of Balmer line at each points of the given :math:`R, Z` grid.
        The shape of array is :math:`(N_R, N_Z)`, where :math:`N_R` and :math:`N_Z` are the number
        of :math:`R` and :math:`Z` grid points respectively.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from cherab.phix.plasma import emission_hydrogen_balmer
        >>> r_grid = [0.3]
        >>> z_grid = [0.0, 0.05]
        >>> emissivity = emission_hydrogen_balmer(r_grid, z_grid)
        ✅ loading plasma ... (data from: phix10)

        >>> emissivity
        array([[13326.92469592, 10161.62601819]])
    """
    HALPHA = (654, 658)
    HBETA = (484, 488)
    HGAMMA = (432, 436)
    HDELTA = (408, 412)

    # validate input arguments
    if isinstance(r_grid, (list, tuple, np.ndarray)):
        r_grid = np.array(r_grid)
    else:
        raise TypeError("r_grid must be an iterable object.")

    if isinstance(z_grid, Iterable):
        z_grid = np.array(z_grid)
    else:
        raise TypeError("z_grid must be an iterable object.")

    if r_grid.ndim != 1 or z_grid.ndim != 1:
        raise ValueError("r_grid and z_grid must be 1D array.")

    if spectrum_step <= 0:
        raise ValueError("spectrum_step must be positive.")

    # obtaine plasma emission models
    world = World()
    plasma, _ = load_plasma(world, eq_model=eq_model)
    models = [i for i in plasma.models]

    if balmer == "alpha":
        spectrum_range = HALPHA
        model = [models[1], models[5]]
    elif balmer == "beta":
        spectrum_range = HBETA
        model = [models[2], models[6]]
    elif balmer == "gamma":
        spectrum_range = HGAMMA
        model = [models[3], models[7]]
    elif balmer == "delta":
        spectrum_range = HDELTA
        model = [models[4], models[8]]
    else:
        raise ValueError("balmer must be one of 'alpha', 'beta', 'gamma', 'delta'")

    # set spectrum params
    min_wavelength, max_wavelength = spectrum_range
    spectrum_bins = round((max_wavelength - min_wavelength) / spectrum_step) + 1

    emissivity = np.zeros((r_grid.size, z_grid.size))

    # sample emissivity
    for i, r in enumerate(r_grid):
        for j, z in enumerate(z_grid):
            for model in models:
                emissivity[i, j] += (
                    model.emission(
                        Point3D(r, 0, z),
                        Vector3D(0, 1, 0),
                        Spectrum(min_wavelength, max_wavelength, spectrum_bins),
                    ).total()
                    * 4.0
                    * np.pi
                )  # [W/m^3/sr] -> [W/m^3]

    return emissivity
