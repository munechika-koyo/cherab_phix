"""Module to offer an helper function to load a plasma object."""
from __future__ import annotations

from collections.abc import Iterable

from raysect.core import Node, translate
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

__all__ = ["load_plasma"]


def load_plasma(
    parent: Node, eq_model: str = "phix10", species: object | None = None
) -> tuple[Plasma, EFITEquilibrium]:
    """Helper function of generating PHiX plasma.

    The plasma model is constructed by the plasma shape from the equilibrium data, the particle
    species data including each particle's density and temperature profile which was estimated from
    the experiment data, and the emission model of bremsstrahlung and line emission (H-alpha,
    H-beta, H-gamma, and H-delta).

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
