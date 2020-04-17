# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from raysect.optical import rotate_x, rotate_z
from raysect.optical.material import AbsorbingSurface, RoughConductor
from raysect.primitive.mesh import Mesh

from cherab.phix.machine.material import RoughSUS316L
from cherab.phix.machine.material import PCTFE


CADMESH_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "geometry", "data")
SUS_ROUGHNESS = 0.0125
# transform ZYX axis to YXZ coordinate
ZXY_to_XYZ = rotate_z(90) * rotate_x(90)

# Component
# [path, matelial, copy times, angle offset]
# stenless group
VESSEL = [
    (os.path.join(CADMESH_PATH, "vessel_wall_half_finite.rsm"), RoughSUS316L(SUS_ROUGHNESS), 2, 0)
]
VACCUM_Flange = [
    (os.path.join(CADMESH_PATH, "vaccum_flange.rsm"), RoughSUS316L(SUS_ROUGHNESS), 1, 0)
]
MG_port = [(os.path.join(CADMESH_PATH, "MG_port.rsm"), RoughSUS316L(SUS_ROUGHNESS), 1, 0)]
LIMITER_BOX = [(os.path.join(CADMESH_PATH, "limiter_box.rsm"), RoughSUS316L(0.25), 1, 0)]
LIMITER_225 = [(os.path.join(CADMESH_PATH, "limiter_225.rsm"), RoughSUS316L(0.25), 1, 0)]
FL = [(os.path.join(CADMESH_PATH, "FL_half.rsm"), RoughSUS316L(0.25), 2, -45)]
FBC_up = [(os.path.join(CADMESH_PATH, "FBC_half_up.rsm"), RoughSUS316L(0.25), 2, -45)]
FBC_down = [(os.path.join(CADMESH_PATH, "FBC_half_down.rsm"), RoughSUS316L(0.25), 2, -45)]
RAIL_up = [(os.path.join(CADMESH_PATH, "rail_half_up.rsm"), RoughSUS316L(0.25), 2, 0)]
RAIL_down = [(os.path.join(CADMESH_PATH, "rail_half_down.rsm"), RoughSUS316L(0.25), 2, 0)]
# teflon group
RAIL_con = [(os.path.join(CADMESH_PATH, "rail_connection_half.rsm"), PCTFE(), 2, 0)]
VESSEL_TEFLON = [(os.path.join(CADMESH_PATH, "vessel_teflon_gasket_half.rsm"), PCTFE(), 2, 0)]

# Complete PHiX mesh for Plasma Facing Compornents
# PHiX_MESH = VESSEL + VACCUM_Flange + MG_port + LIMITER_225 + LIMITER_BOX + FL
PHiX_MESH = (
    VESSEL
    + VACCUM_Flange
    + LIMITER_BOX
    + LIMITER_225
    + FL
    + FBC_up
    + FBC_down
    + RAIL_up
    + RAIL_down
    + RAIL_con
    + VESSEL_TEFLON
)


def import_phix_mesh(
    world,
    override_material=None,
    vessel_material=None,
    vaccume_flange=None,
    limiter_box=None,
    limiter_225=None,
    flux_loop=None,
    magnetron_port=None,
    reflection=True,
):
    if reflection is False:
        override_material = AbsorbingSurface()

    mesh = []

    for mesh_item in PHiX_MESH:
        mesh_path, default_material, ncopy, angle_offset = mesh_item
        if override_material is not None:
            material = override_material
        elif vessel_material and isinstance(default_material, RoughConductor):
            material = vessel_material
        elif vaccume_flange and isinstance(default_material, RoughConductor):
            material = vaccume_flange
        elif magnetron_port and isinstance(default_material, RoughConductor):
            material = magnetron_port
        elif limiter_box and isinstance(default_material, RoughConductor):
            material = limiter_box
        elif limiter_225 and isinstance(default_material, RoughConductor):
            material = limiter_225
        elif flux_loop and isinstance(default_material, RoughConductor):
            material = flux_loop
        # elif lambert_material and isinstance(default_material, Lambert):
        #     material = lambert_material
        else:
            material = default_material

        print("importing {}  ...".format(os.path.split(mesh_path)[1]))

        directory, filename = os.path.split(mesh_path)
        mesh_name, ext = filename.split(".")

        temp_mesh = Mesh.from_file(
            mesh_path,
            parent=world,
            # transform +ve Z-axis to up and +ve X-axis to forward
            transform=rotate_z(angle_offset) * ZXY_to_XYZ,
            material=material,
            name=mesh_name,
        )
        mesh.append(temp_mesh)
        angle = 360.0 / ncopy

        for i in range(1, ncopy):  # copies of the master element
            instance = temp_mesh.instance(
                parent=world,
                transform=rotate_z(angle * i + angle_offset) * ZXY_to_XYZ,
                material=material,
                name=mesh_name,
            )
            mesh.append(instance)

    return mesh
