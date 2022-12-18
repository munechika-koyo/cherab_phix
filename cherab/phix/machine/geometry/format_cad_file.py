from __future__ import annotations

from pathlib import Path

from raysect.optical import World
from raysect.primitive import import_stl

GEOMETRY_PATH = Path(__file__).parent.resolve()
STL_PATH_LIST = [stl for stl in (GEOMETRY_PATH / "data" / "STLfiles").glob("*.STL")]
RSM_PATH = GEOMETRY_PATH / "data" / "RSMfiles"

world = World()

for pfc_path in STL_PATH_LIST:
    print(f"transforming {pfc_path.name} into .rsm file")
    mesh = import_stl(pfc_path, scaling=1.0, parent=world)
    mesh.save(RSM_PATH / pfc_path.with_suffix(".rsm").name)
