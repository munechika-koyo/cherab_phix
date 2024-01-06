from contextlib import nullcontext as does_not_raise

import pytest
from raysect.optical import World
from raysect.optical.observer import RGBPipeline2D

from cherab.phix.observer.fast_camera.camera import load_camera


@pytest.mark.parametrize(
    ["path", "expectation"],
    [
        pytest.param(None, does_not_raise(), id="default"),
        pytest.param("not_exist_path", pytest.raises(FileNotFoundError), id="not_exist_path"),
        pytest.param("invalid_file", pytest.raises(OSError), id="invalid_file"),
    ],
)
def test_load_camera(path, expectation):
    with expectation:
        world = World()

        camera = load_camera(world, path_to_calibration=path)
        camera.pipelines = [RGBPipeline2D(display_progress=False)]
        camera.pixels = (2, 2)
        camera.spectral_rays = 1
        camera.spectral_bins = 1
        camera.per_pixel_samples = 10
        camera.lens_samples = 1

        camera.observe()
