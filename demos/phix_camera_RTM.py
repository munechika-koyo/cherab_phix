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

# ------------ obtain cherab_phix path ----------------------
DIR = os.path.abspath(__file__).split(sep="demos")[0]
# -----------------------------------------------------------


# ------------ Define local function ------------------------
def record_header(world, save_dir=None):
    """Save header information about raysect_scene graph

    Parameters
    ----------
    world : raysect object
        world object
    save_dir : str, optional
        save directry, by default None
    """
    # record infromation
    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        result = (
            "--------------------------------------------------------------------------------\n"
            f"camera name              : {world.observers[0].name}\n"
            f"camera pixels            : {world.observers[0].pixels}\n"
            f"camera per pixel samples : {world.observers[0].per_pixel_samples}\n"
            f"camera lens samples      : {world.observers[0].lens_samples}\n"
            f"camera pixel samples     : {world.observers[0].pixel_samples}\n"
            f"camera spectral bins     : {world.observers[0].spectral_bins}\n"
            f"camera wavelength range  : {world.observers[0].min_wavelength}, {world.observers[0].max_wavelength}\n"
            f"primitives material      : {type(world.primitives[1].material)}\n"
            "--------------------------------------------------------------------------------\n"
        )
        f.write(result)
    print(result)


# ---------------------------------------------------------


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
camera.min_wavelength = 655.6
camera.max_wavelength = 656.8
camera.spectral_rays = 1
camera.spectral_bins = rtm.bins
camera.per_pixel_samples = 10
camera.lens_samples = 20
camera.observe()

# save results
dt_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = os.path.join(DIR, "output", "RTM", f"{dt_now}")
os.makedirs(save_dir)

# save header
record_header(world, save_dir=save_dir)
# save matrix
np.save(os.path.join(save_dir, "RTM"), rtp.matrix)

print(f"successfully saved {os.path.join(save_dir, 'RTM')}")
