# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import glob
from raysect.optical import World
from raysect.primitive import import_stl


GEOMETRY_PATH = os.path.dirname(__file__)
STL_PATH_LIST = glob.glob(os.path.join(GEOMETRY_PATH, "data", "STLfiles", "*.STL"))
RSM_PATH = os.path.join(GEOMETRY_PATH, "data", "RSMfiles")

world = World()

for pfc_fname in STL_PATH_LIST:
    print(f"transforming {pfc_fname.split('.STL')[0]} into .rsm file")
    mesh = import_stl(pfc_fname, scaling=1.0, parent=world)
    mesh.save(os.path.join(RSM_PATH, os.path.splitext(os.path.split(pfc_fname)[-1])[0] + ".rsm"))
