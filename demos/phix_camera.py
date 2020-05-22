import os
import numpy as np
import datetime

from raysect.optical import World
from raysect.optical.observer import RGBPipeline2D, PowerPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D
from cherab.phix.plasma import import_plasma
from cherab.phix.machine import import_phix_mesh
from cherab.phix.observer import import_phix_camera

3
# ------------ obtain cherab_phix path ----------------------
DIR = os.path.abspath(__file__).split(sep="demos")[0]
# -----------------------------------------------------------

# generate scene world
world = World()
# import plasma
plasma, eq = import_plasma(world)
# import phix mesh
mesh = import_phix_mesh(world, reflection=True)
# import phix camera
camera = import_phix_camera(world)

# set pipelines and sampler
rgb = RGBPipeline2D(display_unsaturated_fraction=1.0, name="sRGB", display_progress=False)
# power = PowerPipeline2D(display_progress=False, name="power")
camera.pipelines = [rgb]  # , power]
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=10, cutoff=0.05)

# camera.frame_sampler = sampler
pixels = (256, 512)  # [px]
# pixels = (128, 256)  # [px]
camera.pixels = pixels
camera.min_wavelength = 400
camera.max_wavelength = 780
camera.spectral_rays = 1
camera.spectral_bins = 20
camera.per_pixel_samples = 5
camera.lens_samples = 10


# calculate ray-tracing
camera.observe()

# save
time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = os.path.join(DIR, "output", f"phix_camera({time_now})")
rgb.save(filename + ".png")
# np.save(filename + "_power", power.frame.mean)
print(f"successfully saved {filename.split(os.path.sep)[-1]}")
