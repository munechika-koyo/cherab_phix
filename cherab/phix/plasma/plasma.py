from raysect.core import translate
from raysect.primitive import Cylinder, Subtract
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.openadas import OpenADAS
from cherab.core import Plasma, Line, elements
from cherab.core.math import VectorAxisymmetricMapper
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.phix.plasma import TSCEquilibrium, PHiXSpecies
from cherab.phix.machine.wall_outline import VESSEL_WALL


def import_plasma(parent, folder="phix10", species=None):
    """Helper function of generating PHiX plasma
    As emissions, H :math:`\\alpha`, H :math:`\\beta`, H :math:`\\gamma`, H :math:`\\delta` are applied.

    Parameters
    ----------
    parent : :obj:`~raysect.core.scenegraph.node.Node`
        Raysect's scene-graph parent node
    folder : str
        folder name in which TSC data is stored
    species : object , optional
        user-defined species object having composition which is a list of :obj:`~cherab.core.Species` objects
        and electron distribution function attributes,
        by default :py:class:`.PHiXSpecies`

    Returns
    -------
    tuple
        (:obj:`~cherab.core.Plasma`, :obj:`.TSCEquilibrium`)
    """
    print(f"loading plasma (data from: {folder})...")
    # create TSCEquilibrium instance
    eq = TSCEquilibrium(folder=folder)

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
    species = species or PHiXSpecies(equilibrium=eq)
    plasma.composition = species.composition
    plasma.electron_distribution = species.electron_distribution

    # apply emission from plasma
    h_alpha = Line(elements.hydrogen, 0, (3, 2))  # , wavelength=656.279)
    h_beta = Line(elements.hydrogen, 0, (4, 2))  # , wavelength=486.135)
    h_gamma = Line(elements.hydrogen, 0, (5, 2))  # , wavelength=434.0472)
    h_delta = Line(elements.hydrogen, 0, (6, 2))  # , wavelength=410.1734)
    # ciii_777 = Line(
    #     elements.carbon, 2, ("1s2 2p(2P°) 3d 1D°", " 1s2 2p(2P°) 3p  1P")
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

    return (plasma, eq)


# For debugging
if __name__ == "__main__":
    from raysect.core import World

    world = World()
    plasma, eq = import_plasma(world)
    print([i for i in plasma.models])
    pass
