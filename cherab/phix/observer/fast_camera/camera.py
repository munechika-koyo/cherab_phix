"""Module to offer helper function to load fast camera installed in phix."""
from importlib.resources import files
from pathlib import Path

import numpy as np
from calcam import Calibration
from raysect.optical import AffineMatrix3D, Node

from ...tools import Spinner
from ..thin_lens_ccd import ThinLensCCDArray

__all__ = ["import_phix_camera"]

# Default path to calcam calibration data
CALCAM_PATH = files("cherab.phix.observer.fast_camera.data.calibration").joinpath(
    "shot_17393_ideal.ccc"
)


def import_phix_camera(
    parent: Node,
    path_to_calibration: str | Path | None = None,
):
    """Importing PHiX fast lens camera configured by defalut camera parameters.

    Default camera extrinsic matrix (rotation matrix and translation vector) is loaded from
    ``cherab/phix/observer/fast_camera/data/calibration/shot_17393_ideal.ccc``.
    This file is created by :obj:`calcam` package (See: https://github.com/euratom-software/calcam)

    Parameters
    ----------
    parent
        Raysect's scene-graph parent node
    path_to_calibration
        path to calcam calibration data, by default using
        ``cherab/phix/observer/fast_camera/data/calibration/shot_17393_ideal.ccc``

    Returns
    -------
    :py:class:`.ThinLensCCDArray`
        instance of ThinLensCCDArray object

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.optical import World
        >>> from cherab.phix.observer import import_phix_camera
        >>>
        >>> world = World()
        >>> camera = import_phix_camera(world)
        ✅ importing PHiX camera...
    """
    with Spinner("importing PHiX camera...") as sp:
        # Load calibration data from calcam file
        if path_to_calibration is None:
            cam = Calibration(str(CALCAM_PATH))
        else:
            cam = Calibration(str(path_to_calibration))

        # Get camera rotation matrix and translation vector
        rotation_matrix = cam.get_cam_to_lab_rotation()
        camera_pos = cam.get_pupilpos(coords="Original")

        # Set camera extrinsic matrix
        transform = AffineMatrix3D(
            np.block([[rotation_matrix, camera_pos.reshape(3, 1)], [np.array([0, 0, 0, 1])]])
        )

        # === generate ThinLensCCDArray object ===
        camera = ThinLensCCDArray(
            pixels=(256, 512),
            width=25.6e-3 * 256 / 1280,
            focal_length=10.0e-3,
            working_distance=50.0e-2,
            f_number=0 * (22 - 3.5) / 10 + 3.5,
            parent=parent,
            pipelines=None,
            transform=transform,
            name="PHiX fast-visible camera",
        )
        sp.ok()

    return camera
