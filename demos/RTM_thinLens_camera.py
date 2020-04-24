import os
import numpy as np
import datetime

from raysect.optical import World
from raysect.optical.observer import FullFrameSampler2D
from cherab.tools.raytransfer import RayTransferPipeline2D
from cherab.phix.observer import import_phix_camera
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.machine import import_phix_mesh
from cherab.phix.tools import import_phix_rtm


# generate scene world
world = World()
# import phix equilibrium
eq = TSCEquilibrium()
# import phix machine configuration
mesh = import_phix_mesh(world, reflection=True)
# import phix raytrasfer matrix
rtm = import_phix_rtm(world, equilibrium=eq)
# import phix camera
camera = import_phix_camera(world)

# set camera's parameters
rtp = RayTransferPipeline2D()
camera.pipelines = [rtp]

pixels = (128, 256)  # [px]
camera.pixels = pixels
# calculate rtm for only (w, h) = (64, 128) element
# focus_pixel = (64, 128)
# mask = np.zeros(pixels, dtype=np.bool)
# mask[focus_pixel[0] - 1, focus_pixel[1] - 1] = True
camera.frame_sampler = FullFrameSampler2D()
camera.min_wavelength = 650
camera.max_wavelength = 660
camera.spectral_rays = 1
camera.spectral_bins = rtm.bins
camera.per_pixel_samples = 10
camera.lens_samples = 20
camera.observe()

# save results
dt_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = os.path.join("..", "output", f"thin_lens_camera({dt_now})")
np.save(filename + "_power", rtp.matrix)
# power.save(filename + "_power")
print(f"successfully saved {filename.split(os.path.sep)[-1]}")
