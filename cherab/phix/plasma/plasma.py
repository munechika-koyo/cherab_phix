from raysect.core import World, translate, Vector3D
from raysect.primitive import Cylinder, Subtract
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.openadas import OpenADAS
from cherab.core import Plasma
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.machine.wall_outline import VESSEL_WALL


def import_plasma(parent=World()):
    """Helper function of generating PHiX plasma

    Parameters
    ----------
    parent : Node, optional
        Raysect's scene-graph parent node, by default World()
    """
    # create EFIT objects from TSCEquilibrium class
    equi = TSCEquilibrium().create_EFIT()

    # create atomic data source
    adas = OpenADAS(permit_extrapolation=True)

    # generate plasma object instance
    plasma = Plasma(parent=parent, name="PHiX_plasma")

    # setting plasma properties
    plasma.atomic_data = adas
    plasma.integrator = NumericalIntegrator()

    # create plasma geometry as subtraction of two cylinders
    inner_radius = VESSEL_WALL[:, 0].min()
    outer_radius = VESSEL_WALL[:, 0].max()
    height = VESSEL_WALL[:, 1].max() - VESSEL_WALL[:, 1].min()

    inner_cylinder = Cylinder(inner_radius, height)
    outer_cylinder = Cylinder(outer_radius, height)

    plasma.geometry = Subtract(outer_cylinder, inner_cylinder)
    plasma.geometry_transform = translate(0, 0, VESSEL_WALL[:, 1].min())
