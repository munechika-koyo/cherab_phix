# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from raysect.optical import rotate
from raysect.optical.material import AbsorbingSurface, RoughConductor
from raysect.primitive.mesh import Mesh

from cherab.phix.machine.material import RoughSUS316L


CADMESH_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "geometry", "data")

# Component
# [path, matelial, copy times, angle offset]
VESSEL = [(os.path.join(CADMESH_PATH, "vessel_wall_half.rsm"), RoughSUS316L(0.0625), 1, 0)]

# Complete PHiX mesh for Plasma Facing Compornents
PHiX_MESH = VESSEL


def import_phix_mesh(world, override_material=None, vessel_material=None, reflection=True):
    if reflection is False:
        override_material = AbsorbingSurface()

    mesh = []

    for mesh_item in PHiX_MESH:
        mesh_path, default_material, ncopy, angle_offset = mesh_item
        if override_material is not None:
            material = override_material
        elif vessel_material and isinstance(default_material, RoughConductor):
            material = vessel_material
        # elif beryllium_material and isinstance(default_material, RoughBeryllium):
        #     material = beryllium_material
        # elif iron_material and isinstance(default_material, RoughIron):
        #     material = iron_material
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
            transform=rotate(
                90, 0, 90 + angle_offset
            ),  # transform +ve Z-axis to up and +ve X-axis to forward
            material=material,
            name=mesh_name,
        )
        mesh.append(temp_mesh)
        angle = 360.0 / ncopy

        for i in range(1, ncopy):  # copies of the master element
            instance = temp_mesh.instance(
                parent=world,
                transform=rotate(0, 0, angle * i + angle_offset),
                material=material,
                name=mesh_name,
            )
            mesh.append(instance)

    return mesh
