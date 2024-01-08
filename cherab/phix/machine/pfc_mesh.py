"""Module to offer helper function to load plasma facing component meshes."""
from __future__ import annotations

from collections import defaultdict
from importlib.resources import files

from plotly import graph_objects as go
from plotly.graph_objects import Figure
from raysect.optical import World, rotate_z
from raysect.optical.material import AbsorbingSurface, Material
from raysect.primitive.mesh import Mesh
from scipy.spatial.transform import Rotation

from ..tools.spinner import Spinner
from .material import PCTFE, RoughSUS316L

__all__ = ["load_pfc_mesh", "show_PFCs_3D"]

# Path to directory containing .rsm files
RSM_DIR = files("cherab.phix.machine.geometry.data.RSMfiles")

# TODO: omtimization of roughness
SUS_ROUGHNESS = 0.0125

# List of Plasma Facing Components (filename is "**.rsm")
COMPONENTS: dict[str, tuple[str, Material]] = {
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

# How many times each PFC element must be copy-pasted in toroidal direction
NCOPY: dict[str, int] = defaultdict(lambda: 1)
NCOPY["Vaccum Vessel"] = 2
NCOPY["Flux Loop"] = 2
NCOPY["Feed Back Coil (upper)"] = 2
NCOPY["Feed Back Coil (lower)"] = 2
NCOPY["Rail (upper)"] = 2
NCOPY["Rail (lower)"] = 2
NCOPY["Rail Connection"] = 2
NCOPY["Vacuum Vessel Gasket"] = 2

# Offset toroidal angle
ANG_OFFSET: dict[str, float] = defaultdict(lambda: 0.0)
ANG_OFFSET["Flux Loop"] = 0.0  # deg


def load_pfc_mesh(
    world: World,
    override_materials: dict[str, Material] | None = None,
    reflection: bool = True,
    is_fine_mesh: bool = True,
) -> dict[str, list[Mesh]]:
    """Load plasma facing component meshes.

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
    is_fine_mesh
        whether or not to use fine mesh for the vaccum vessel, by default True.

    Returns
    -------
    dict[str, list[:obj:`~raysect.primitive.mesh.mesh.Mesh`]]
        containing mesh name and :obj:`~raysect.primitive.mesh.mesh.Mesh` objects

    Examples
    --------
    .. prompt:: python

        from raysect.optical import World
        from cherab.phix.machine import load_pfc_mesh

        world = World()
        meshes = load_pfc_mesh(world, reflection=True)
    """
    if is_fine_mesh:
        COMPONENTS["Vaccum Vessel"] = ("vessel_wall_fine", RoughSUS316L(SUS_ROUGHNESS))
    else:
        COMPONENTS["Vaccum Vessel"] = ("vessel_wall", RoughSUS316L(SUS_ROUGHNESS))

    meshes = {}

    with Spinner(text="Loading PFCs...") as spinner:
        for mesh_name, (filename, default_material) in COMPONENTS.items():
            try:
                spinner.text = f"Loading {mesh_name}..."

                # === set material ===
                if not reflection:
                    material = AbsorbingSurface()
                else:
                    if isinstance(override_materials, dict):
                        material = override_materials.get(mesh_name, None)
                        if material is None:
                            material = default_material
                        elif isinstance(material, Material):
                            pass
                        else:
                            raise TypeError(
                                f"override_materials[{mesh_name}] must be Material instance."
                            )
                    elif override_materials is None:
                        material = default_material
                    else:
                        raise TypeError(
                            f"override_materials must be dict[str, Material] instance or None. ({mesh_name})"
                        )

                # === load mesh ===
                # master element
                meshes[mesh_name] = [
                    Mesh.from_file(
                        RSM_DIR / f"{filename}.rsm",
                        parent=world,
                        transform=rotate_z(ANG_OFFSET[mesh_name]),
                        material=material,
                        name=f"{mesh_name} 1" if NCOPY[mesh_name] > 1 else f"{mesh_name}",
                    )
                ]

                # copies of the master element
                angle = 360.0 / NCOPY[mesh_name]
                for i in range(1, NCOPY[mesh_name]):
                    meshes[mesh_name].append(
                        meshes[mesh_name][0].instance(
                            parent=world,
                            transform=rotate_z(angle * i + ANG_OFFSET[mesh_name]),
                            material=material,
                            name=f"{mesh_name} {i + 1}",
                        )
                    )

                # === print result ===
                material_str = str(material).split()[0].split(".")[-1]
                if roughness := getattr(material, "roughness", None):
                    material_str = f"{material_str: <12} (roughness: {roughness:.4f})"
                else:
                    material_str = f"{material_str}"
                spinner.write(f"✅ {mesh_name: <22}: {material_str}")

            except Exception as e:
                spinner.write(f"💥 {e}")

    return meshes


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
    :obj:`~plotly.graph_objects.Figure`
        plotly Figure object

    Examples
    --------
    .. prompt:: python

        fig = show_PFCs_3D(fig_size=(700, 500))
        fig.show()

    The above codes automatically lauch a browser to show the figure when it is executed in
    the python interpreter like the following picture:

    .. image:: ../_static/images/show_PFCs_3D_example.png
    """
    if fig is None or not isinstance(fig, Figure):
        fig = go.Figure()

    # load meshes
    world = World()
    meshes = load_pfc_mesh(world, reflection=False, is_fine_mesh=False)

    for _, mesh_list in meshes.items():
        for mesh in mesh_list:
            # Rotate mesh by its transform matrix
            transform = mesh.to_root()
            r = Rotation.from_matrix([[transform[i, j] for j in range(3)] for i in range(3)])
            x, y, z = r.apply(mesh.data.vertices).T
            i, j, k = mesh.data.triangles.T

            # Create Mesh3d object
            mesh3D = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=True,
                colorscale=[[0, "#e5dee5"], [1, "#e5dee5"]],
                intensity=z,
                name=f"{mesh.name}",
                text=f"{mesh.name}",
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
                hovertemplate=f"<b>{mesh.name}</b><br>" + "x: %{x}<br>y: %{y}<br>z: %{z}<br>"
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
