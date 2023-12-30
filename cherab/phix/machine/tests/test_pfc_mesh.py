from collections import defaultdict

import pytest
from plotly import graph_objects as go
from raysect.optical import World
from raysect.optical.library import RoughAluminium
from raysect.optical.material import AbsorbingSurface

from cherab.phix.machine.material import PCTFE, RoughSUS316L
from cherab.phix.machine.pfc_mesh import load_pfc_mesh, show_PFCs_3D

reflection_materials = defaultdict(lambda: RoughSUS316L)
reflection_materials["Rail Connection"] = PCTFE
reflection_materials["Vacuum Vessel Gasket"] = PCTFE

absorb_materials = defaultdict(lambda: AbsorbingSurface)

override_materials = defaultdict(lambda: RoughSUS316L)
override_materials["Vessel Wall"] = RoughAluminium


@pytest.mark.parametrize(
    ["reflection", "override_materials", "default_material"],
    [
        pytest.param(True, None, reflection_materials, id="reflection"),
        pytest.param(False, None, absorb_materials, id="no reflection"),
        pytest.param(
            True,
            {"Vesell Wall", RoughAluminium(0.1)},
            override_materials,
            id="override materials",
        ),
    ],
)
def test_load_pfc_mesh(reflection, override_materials, default_material):
    world = World()
    meshes = load_pfc_mesh(world, reflection=reflection, override_materials=override_materials)

    for mesh_name, mesh_list in meshes.items():
        for mesh in mesh_list:
            assert isinstance(mesh.material, default_material[mesh_name])


def test_show_PFCs_3D():
    fig = show_PFCs_3D()
    assert isinstance(fig, go.Figure)
