# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from raysect.optical import rotate_x, rotate_z
from raysect.optical.material import AbsorbingSurface, RoughConductor, Material
from raysect.primitive.mesh import Mesh

from cherab.phix.machine.material import RoughSUS316L
from cherab.phix.machine.material import PCTFE


CADMESH_PATH = os.path.join(os.path.dirname(__file__), "geometry", "data", "RSMfiles")
# TODO: omtimization of roughness
SUS_ROUGHNESS = 0.0125

# Component
# [name, path, matelial, copy times, angle offset]
# stenless group
VESSEL = [("Vacuum Vessel", os.path.join(CADMESH_PATH, "vessel_wall_finite.rsm"), RoughSUS316L(SUS_ROUGHNESS), 2, 0)]
VACCUM_Flange = [("Vacuum Flange", os.path.join(CADMESH_PATH, "vaccum_flange.rsm"), RoughSUS316L(SUS_ROUGHNESS), 1, 0)]
MG_port = [("Magnetron Port", os.path.join(CADMESH_PATH, "MG_port.rsm"), RoughSUS316L(SUS_ROUGHNESS), 1, 0)]
LIMITER_BOX = [("Limiter Box", os.path.join(CADMESH_PATH, "limiter_box.rsm"), RoughSUS316L(0.25), 1, 0)]
LIMITER_225 = [("Limiter 225", os.path.join(CADMESH_PATH, "limiter_225.rsm"), RoughSUS316L(0.25), 1, 0)]
FL = [("Flux Loop", os.path.join(CADMESH_PATH, "FL_half.rsm"), RoughSUS316L(0.25), 2, -45)]
FBC_up = [("Feed Back Coil", os.path.join(CADMESH_PATH, "FBC_half_up.rsm"), RoughSUS316L(0.25), 2, 0)]
FBC_down = [("Feed Back Coil", os.path.join(CADMESH_PATH, "FBC_half_down.rsm"), RoughSUS316L(0.25), 2, 0)]
RAIL_up = [("Rail", os.path.join(CADMESH_PATH, "rail_half_up.rsm"), RoughSUS316L(0.25), 2, 0)]
RAIL_down = [("Rail", os.path.join(CADMESH_PATH, "rail_half_down.rsm"), RoughSUS316L(0.25), 2, 0)]
# teflon group
RAIL_con = [("Rail connection", os.path.join(CADMESH_PATH, "rail_connection_half.rsm"), PCTFE(), 2, 0)]
VESSEL_GASKET = [("Vacuum Vessel Gasket", os.path.join(CADMESH_PATH, "vessel_gasket_half.rsm"), PCTFE(), 2, 0)]

# Complete PHiX mesh for Plasma Facing Compornents
PHiX_MESH = (
    VESSEL
    + VACCUM_Flange
    + MG_port
    + LIMITER_BOX
    + LIMITER_225
    + FL
    + FBC_up
    + FBC_down
    + RAIL_up
    + RAIL_down
    + RAIL_con
    + VESSEL_GASKET
)


def import_phix_mesh(
    world,
    override_material=None,
    vacuum_vessel=None,
    vaccum_flange=None,
    magnetron_port=None,
    limiter_box=None,
    limiter_225=None,
    flux_loop=None,
    feed_back_coil=None,
    rail=None,
    rail_connection=None,
    vessel_gasket=None,
    reflection=True,
):
    """Import PHiX Mesh files (.rsm) from directory data/RSMfiles
    Each Meshes allow the user to use an user-defined material.

    Parameters
    ----------
    world : Raysect object
        scene-graph node object, here World() should be put.
    override_material : Material, optional
        user-defined Material is applied to all of meshes, by default None
    vacuum_vessel : Material, optional
        user-defined Material applied to Vacuum Vessel, by default None
    vaccum_flange : Material, optional
        user-defined Material applied to Vacuum Vessel Flange, by default None
    magnetron_port : Material, optional
        user-defined Material applied to Magnetron Port, by default None
    limiter_box : Material, optional
        user-defined Material applied to Limiter Box, by default None
    limiter_225 : Material, optional
        user-defined Material applied to Limiter at 22.5 degree, by default None
    flux_loop : Material, optional
        user-defined Material applied to Flux Loop, by default None
    feed_back_coil : Material, optional
        user-defined Material applied to Feed Back Coil, by default None
    rail : Material, optional
        user-defined Material applied to Rail, by default None
    rail_connection : Material, optional
        user-defined Material applied to Rail Connection, by default None
    vessel_gasket : Material, optional
        user-defined Material applied to Vaccum Vessel Gasket, by default None
    reflection : bool, optional
        whether or not to consider reflection light, by default True
        If reflection == False, all of meshes' material is replaced to AbsorbingSurface()

    Returns
    -------
    list
        list containing mesh objects
    """

    if reflection is False:
        override_material = AbsorbingSurface()

    mesh = []

    for mesh_item in PHiX_MESH:
        mesh_name, mesh_path, default_material, ncopy, angle_offset = mesh_item
        if override_material is not None and isinstance(override_material, Material):
            material = override_material
        elif vacuum_vessel is not None and mesh_name == "Vacuum Vessel" and isinstance(vacuum_vessel, Material):
            material = vacuum_vessel
        elif vaccum_flange is not None and mesh_name == "Vacuum Flange" and isinstance(vaccum_flange, Material):
            material = vaccum_flange
        elif magnetron_port is not None and mesh_name == "Magnetron Port" and isinstance(magnetron_port, Material):
            material = magnetron_port
        elif limiter_box is not None and mesh_name == "Limiter Box" and isinstance(limiter_box, Material):
            material = limiter_box
        elif limiter_225 is not None and mesh_name == "Limiter 225" and isinstance(limiter_225, Material):
            material = limiter_225
        elif flux_loop is not None and mesh_name == "Flux Loop" and isinstance(flux_loop, Material):
            material = flux_loop
        elif feed_back_coil is not None and mesh_name == "Feed Back Coil" and isinstance(feed_back_coil, Material):
            material = feed_back_coil
        elif rail is not None and mesh_name == "Rail" and isinstance(rail, Material):
            material = rail
        elif rail_connection is not None and mesh_name == "Rail connection" and isinstance(rail_connection, Material):
            material = rail_connection
        elif vessel_gasket is not None and mesh_name == "Vacuum Vessel Gasket" and isinstance(vessel_gasket, Material):
            material = vessel_gasket
        else:
            material = default_material

        # directory, filename = os.path.split(mesh_path)
        # mesh_name, ext = filename.split(".")

        roughness = getattr(material, "roughness", None)

        print(f'importing {mesh_name}, {str(material).split(" ")[0].split(".")[-1]}, roughness: {roughness}')

        temp_mesh = Mesh.from_file(
            mesh_path, parent=world, transform=rotate_z(angle_offset), material=material, name=mesh_name
        )
        mesh.append(temp_mesh)
        angle = 360.0 / ncopy

        for i in range(1, ncopy):  # copies of the master element
            instance = temp_mesh.instance(
                parent=world, transform=rotate_z(angle * i + angle_offset), material=material, name=mesh_name
            )
            mesh.append(instance)

    return mesh


# debug
if __name__ == "__main__":

    from raysect.optical import World

    world = World()
    mesh = import_phix_mesh(world)

    print("test")
