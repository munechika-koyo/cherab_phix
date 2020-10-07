import os
import numpy as np
import datetime

from raysect.optical import World
from raysect.optical.observer import SpectralPowerPipeline2D, PowerPipeline2D
from raysect.optical.observer import SpectralAdaptiveSampler2D
from raysect.optical.observer import FullFrameSampler2D
from cherab.phix.plasma import import_plasma
from cherab.phix.machine import import_phix_mesh
from cherab.phix.observer import import_phix_camera


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
# import plasma
plasma, eq = import_plasma(world)
# import phix mesh
mesh = import_phix_mesh(world, reflection=True)
# import phix camera
calib_data = "shot_17393"
camera = import_phix_camera(world)
camera.name = "PHiX fast-visible camera (calibration: " + calib_data + ")"

# set pipelines and sampler
sp = SpectralPowerPipeline2D(name="Spectrul_Power")
pw = PowerPipeline2D(name="Power", display_progress=False, display_auto_exposure=False)
camera.pipelines = [sp, pw]

# camera.frame_sampler = sampler
pixels = (256, 512)  # [px]
# pixels = (128, 256)  # [px]

# mask = np.zeros(pixels, dtype=np.bool)
# focus_pixel = (128, 256)
# mask[focus_pixel[0] - 1, focus_pixel[1] - 1] = True
# camera.frame_sampler = FullFrameSampler2D(mask)
# camera.frame_sampler = FullFrameSampler2D()
sampler = SpectralAdaptiveSampler2D(sp, ratio=10, fraction=0.2, min_samples=50, cutoff=0.05, reduction_method="mean")
camera.frame_sampler = sampler
camera.pixels = pixels
camera.spectral_rays = 1
camera.spectral_bins = 50
camera.per_pixel_samples = 5
camera.lens_samples = 10
# camera.per_pixel_samples = 10
# camera.lens_samples = 20
# iteration of three wavelength range
Halpha = (655.6, 656.8)
Hbeta = (485.6, 486.5)
Hgamma = (433.6, 434.4)
# H_alpha_peak = 656.0
# H_beta_peak = 486.0
# H_gamma_peak = 434.0

for rage_wavelength in [Halpha]:

    camera.min_wavelength = rage_wavelength[0]
    camera.max_wavelength = rage_wavelength[1]

    # calculate ray-tracing
    camera.observe()

    # save
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(DIR, "output", "synthetic_data", f"{time_now}")
    os.makedirs(save_dir)

    # save header
    record_header(world, save_dir=save_dir)

    # save wavelength
    np.save(os.path.join(save_dir, "wavelengths"), sp.wavelengths)

    # save spectral power
    np.save(os.path.join(save_dir, "SP_frame_mean"), sp.frame.mean)
    np.save(os.path.join(save_dir, "SP_frame_error"), sp.frame.variance)

    # save Power in range(min, max_wavelength)
    np.save(os.path.join(save_dir, "Power"), pw.frame.mean)
