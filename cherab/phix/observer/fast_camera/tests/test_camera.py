import pytest
from raysect.optical import World
from raysect.optical.observer import RGBPipeline2D

from cherab.phix.observer.fast_camera.camera import import_phix_camera


def test_import_phix_camera():
    world = World()

    camera = import_phix_camera(world, path_to_calibration=None)
    camera.pipelines = [RGBPipeline2D(display_progress=False)]
    camera.pixels = (2, 2)
    camera.spectral_rays = 1
    camera.spectral_bins = 1
    camera.per_pixel_samples = 10
    camera.lens_samples = 1

    camera.observe()

    with pytest.raises(OSError):
        import_phix_camera(world, path_to_calibration="invalid_path")
