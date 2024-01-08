"""Module to convert .STL files into .RSM files."""
from __future__ import annotations

from pathlib import Path

from raysect.optical import World
from raysect.primitive import import_stl

__all__ = ["stl_to_rsm"]


GEOMETRY_PATH = Path(__file__).parent.resolve()
STL_PATH_LIST = [stl for stl in (GEOMETRY_PATH / "data" / "STLfiles").glob("*.STL")]
RSM_PATH = GEOMETRY_PATH / "data" / "RSMfiles"


def stl_to_rsm(stl_path: Path | str, save_dir: Path | str) -> None:
    """Converts .STL files into .RSM files.

    The file name of the .RSM file will be the same as the .STL file,
    and the converting scale is set to 1.0.

    Parameters
    ----------
    stl_path
        Path to the .STL file.
    save_dir
        Path to the directory where the .RSM file will be saved.
    """
    # validate arguments
    stl_path = Path(stl_path)
    save_dir = Path(save_dir)
    if not stl_path.exists():
        raise FileNotFoundError(f"{stl_path} does not exist")

    if not save_dir.exists():
        raise FileNotFoundError(f"{save_dir} does not exist")

    world = World()

    print(f"transforming {stl_path.name} into .rsm file")
    mesh = import_stl(stl_path, scaling=1.0, parent=world)
    mesh.save(save_dir / stl_path.with_suffix(".rsm").name)


if __name__ == "__main__":
    for stl_path in STL_PATH_LIST:
        stl_to_rsm(stl_path, RSM_PATH)
