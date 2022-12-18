"""Module to offer helper function to load plasma facing component meshes."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from plotly import graph_objects as go
from plotly.graph_objects import Figure
from raysect.optical import World, rotate_z
from raysect.optical.material import AbsorbingSurface, Material
from raysect.primitive.mesh import Mesh
from scipy.spatial.transform import Rotation
from stl.base import BaseMesh
from stl.mesh import Mesh as STLMesh

from cherab.phix.machine.material import PCTFE, RoughSUS316L
from cherab.phix.tools.spinner import Spinner

__all__ = ["import_phix_mesh", "show_PFCs_3D"]


RSM_DIR = Path(__file__).parent.resolve() / "geometry" / "data" / "RSMfiles"
STL_DIR = Path(__file__).parent.resolve() / "geometry" / "data" / "STLfiles"

# TODO: omtimization of roughness
SUS_ROUGHNESS = 0.0125

COMPONENTS = {
    # name: (filename, material object)
    "Vaccum Vessel": ("vessel_wall_fine", RoughSUS316L(SUS_ROUGHNESS)),
    "Vacuum Flange": ("vaccum_flange", RoughSUS316L(SUS_ROUGHNESS)),
    "Magnetron Port": ("MG_port", RoughSUS316L(SUS_ROUGHNESS)),
    "Limiter Box": ("limiter_box", RoughSUS316L(0.25)),
    "Limiter 225": ("limiter_225", RoughSUS316L(0.25)),
    "Flux Loop": ("FL_half", RoughSUS316L(0.25)),
    "Feed Back Coil (upper)": ("FBC_half_up", RoughSUS316L(0.25)),
    "Feed Back Coil (lower)": ("FBC_half_down", RoughSUS316L(0.25)),
    "Rail (upper)": ("rail_half_up", RoughSUS316L(0.25)),
    "Rail (lower)": ("rail_half_down", RoughSUS316L(0.25)),
    "Rail Connection": ("rail_connection_half", PCTFE()),
    "Vacuum Vessel Gasket": ("vessel_gasket_half", PCTFE()),
}

NCOPY = defaultdict(lambda: 1)
NCOPY["Vaccum Vessel"] = 2
NCOPY["Flux Loop"] = 2
NCOPY["Feed Back Coil (upper)"] = 2
NCOPY["Feed Back Coil (lower)"] = 2
NCOPY["Rail (upper)"] = 2
NCOPY["Rail (lower)"] = 2
NCOPY["Rail Connection"] = 2

ANG_OFFSET = defaultdict(lambda: 0)
ANG_OFFSET["Flux Loop"] = -45


def import_phix_mesh(
    world: World,
    override_materials: dict[str, Material] | None = None,
    reflection: bool = True,
) -> dict[str, list[Mesh]]:
    """Import PHiX Plasma facing component meshes.

    Each Meshes allow the user to use an user-defined material which inherites
    :obj:`~raysect.optical.material.material.Material`.

    Parameters
    ----------
    world
        The world scenegraph belonging to these materials.
    override_materials
        user-defined material. Set up like ``{"Vaccum Vessel": RoughSUS316L(0.05), ...}``.
    reflection
        whether or not to consider reflection light, by default True.
        If ``False``, all of meshes' material are replaced to
        :obj:`~raysect.optical.material.absorber.AbsorbingSurface`

    Returns
    -------
    dict[str, list[:obj:`~raysect.primitive.mesh.mesh.Mesh`]]
        containing mesh name and :obj:`~raysect.primitive.mesh.mesh.Mesh` objects

    Example
    -------
    .. prompt:: python

        from raysect.optical import World
        from cherab.phix.machine import import_phix_mesh

        world = World()
        meshes = import_phix_mesh(world, reflection=True)
    """

    if not reflection:
        override_materials = defaultdict(lambda: AbsorbingSurface())

    mesh = {}

    with Spinner(text="Loading PFCs...") as spinner:
        for mesh_name, (filename, material) in COMPONENTS.items():
            try:
                if override_materials is not None:
                    material = override_materials[mesh_name]

                # master element
                mesh[mesh_name] = [
                    Mesh.from_file(
                        RSM_DIR / f"{filename}.rsm",
                        parent=world,
                        transform=rotate_z(ANG_OFFSET[mesh_name]),
                        material=material,
                        name=mesh_name,
                    )
                ]

                # copies of the master element
                angle = 360.0 / NCOPY[mesh_name]
                for i in range(1, NCOPY[mesh_name]):
                    mesh[mesh_name].append(
                        mesh[mesh_name][0].instance(
                            parent=world,
                            transform=rotate_z(angle * i + ANG_OFFSET[mesh_name]),
                            material=material,
                            name=mesh_name,
                        )
                    )

                material_str = str(material).split()[0].split(".")[-1]
                if roughness := getattr(material, "roughness", None):
                    material_str = f"{material_str} (roughness: {roughness:.4f})"
                else:
                    material_str = f"{material_str}"
                spinner.write(f"✅ {mesh_name}: {material_str}")

            except Exception as e:
                spinner.write(f"💥 {e}")

    return mesh


def show_PFCs_3D(fig: Figure | None = None, fig_size: tuple[int, int] = (700, 500)) -> Figure:
    """Show PHiX Plasma Facing Components in 3-D space.

    Plot 3D meshes of PFCs with plotly.

    Parameters
    ----------
    fig
        plotly Figure object, by default :obj:`~plotly.graph_objects.Figure`.
    fig_size
        figure size, be default (700, 500) px.

    Returns
    -------
        plotly Figure object

    Example
    -------
    .. prompt:: python

        fig = show_PFCs_3D(fig_size=(700, 500))
        fig.show()

    The above codes automatically lauch a browser to show the figure when it is executed in
    the python interpreter like the following picture:

    .. image:: ../_static/images/show_PFCs_3D_example.png
    """
    fig = go.Figure()

    for mesh_name, (filename, _) in COMPONENTS.items():
        # use not fine mesh in vessel
        if mesh_name == "Vaccum Vessel":
            filename = "vessel_wall"

        stl_mesh = STLMesh.from_file(STL_DIR / f"{filename}.STL")
        vertices, I, J, K = _stl2mesh3d(stl_mesh)
        # Offset rotatation
        # rot = Rotation.from_euler("z", ANG_OFFSET[mesh_name], degrees=True)
        # vertices = rot.apply(vertices)
        x, y, z = vertices.T

        mesh3D = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=I,
            j=J,
            k=K,
            flatshading=True,
            colorscale=[[0, "#e5dee5"], [1, "#e5dee5"]],
            intensity=z,
            name=f"{mesh_name}",
            text=f"{mesh_name}",
            showscale=False,
            showlegend=True,
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                fresnel=0.1,
                specular=1,
                roughness=0.1,
                facenormalsepsilon=0,
            ),
            lightposition=dict(x=3000, y=3000, z=10000),
            hovertemplate=f"<b>{mesh_name}</b><br>" + "x: %{x}<br>y: %{y}<br>z: %{z}<br>"
            "<extra></extra>",
        )
        fig.add_trace(mesh3D)

        # copies of the master element
        angle = 360.0 / NCOPY[mesh_name]
        rot = Rotation.from_euler("z", angle, degrees=True)
        for i in range(1, NCOPY[mesh_name]):
            vertices = rot.apply(vertices)
            x, y, z = vertices.T
            mesh3D = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=I,
                j=J,
                k=K,
                flatshading=True,
                colorscale=[[0, "#e5dee5"], [1, "#e5dee5"]],
                intensity=z,
                name=f"{mesh_name} {i + 1}",
                text=f"{mesh_name} {i + 1}",
                showscale=False,
                showlegend=True,
                lighting=dict(
                    ambient=0.18,
                    diffuse=1,
                    fresnel=0.1,
                    specular=1,
                    roughness=0.1,
                    facenormalsepsilon=0,
                ),
                lightposition=dict(x=3000, y=3000, z=10000),
                hovertemplate=f"<b>{mesh_name} {i + 1}</b><br>"
                + "x: %{x}<br>"
                + "y: %{y}<br>"
                + "z: %{z}<br>"
                "<extra></extra>",
            )
            fig.add_trace(mesh3D)

    fig.update_layout(
        paper_bgcolor="rgb(1,1,1)",
        title_text="PHiX device",
        title_x=0.5,
        font_color="white",
        hoverlabel_grouptitlefont_color="black",
        width=fig_size[0],
        height=fig_size[1],
        scene_aspectmode="data",
        margin=dict(r=10, l=10, b=10, t=35),
        scene_xaxis_visible=False,
        scene_yaxis_visible=False,
        scene_zaxis_visible=False,
    )

    return fig


def _stl2mesh3d(stl_mesh: BaseMesh) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Extracts the unique vertices and the lists I, J, K to define a Plotly
    mesh3d.

    Parameters
    ----------
    stl_mesh
        loaded by numpy-stl module from a stl file

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        2D-array of vertices
        A vector of vertex indices, representing the "first" vertex of triangle
        A vector of vertex indices, representing the "second" vertex of triangle
        A vector of vertex indices, representing the "third" vertex of triangle
    """
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    # the array stl_mesh.vectors.reshape(p * q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3 * k for k in range(p)])
    J = np.take(ixr, [3 * k + 1 for k in range(p)])
    K = np.take(ixr, [3 * k + 2 for k in range(p)])
    return (vertices, I, J, K)


# debug
if __name__ == "__main__":
    world = World()
    mesh = import_phix_mesh(world, reflection=True)
    # fig = show_PFCs_3D()
    # fig.show()
