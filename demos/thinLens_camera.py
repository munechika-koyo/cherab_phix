import numpy as np
from matplotlib import pyplot as plt, cm, colors
import datetime

from raysect.optical import World
from raysect.optical.observer import RGBPipeline2D, PowerPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D, FullFrameSampler2D
from cherab.tools.raytransfer import RayTransferPipeline2D
from cherab.phix.observer import import_phix_camera
from cherab.phix.plasma import import_plasma
from cherab.phix.machine import import_phix_mesh


# generate scene world
world = World()
# import plasma
plasma, eq = import_plasma(world)
# import phix machine configuration
mesh = import_phix_mesh(world, reflection=True)
# import phix camera
camera = import_phix_camera(world)


# set camera's parameters
rgb = RGBPipeline2D(display_unsaturated_fraction=1.0, name="sRGB", display_progress=False)
power = PowerPipeline2D(display_progress=False, name="power")
# rtp = RayTransferPipeline2D()
camera.pipelines = [rgb, power]

pixels = (128, 256)  # [px]
camera.pixels = pixels
# calculate rtm for only (w, h) = (64, 128) element
# focus_pixel = (64, 128)
# mask = np.zeros(pixels, dtype=np.bool)
# mask[focus_pixel[0] - 1, focus_pixel[1] - 1] = True
# camera.frame_sampler = FullFrameSampler2D(mask)
camera.min_wavelength = 650
camera.max_wavelength = 660
camera.spectral_rays = 1
camera.spectral_bins = 1
camera.per_pixel_samples = 10
camera.lens_samples = 10
camera.F_value = 3.5
camera.observe()

filename = f"../output/thin_lens_camera({datetime.datetime.now()})"
rgb.save(filename + ".png")
np.save(filename, power)
power.save(filename + "_power")
