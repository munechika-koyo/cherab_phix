# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import glob
from raysect.optical import World
from raysect.primitive import import_stl


CADMESH_PATH = os.path.abspath(os.path.dirname(__file__))
PFC_PATH_LIST = glob.glob(os.path.join(CADMESH_PATH, "data/*.STL"))

world = World()

for pfc_fname in PFC_PATH_LIST:
    print(f"transforming {pfc_fname.split('.STL')[0]} into .rsm file")
    mesh = import_stl(pfc_fname, scaling=1.0e-3, parent=world)  # [mm] -> [m]
    mesh.save(pfc_fname.split(".STL")[0] + ".rsm")
