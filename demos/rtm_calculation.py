"""
Calculate RayTransfer Matrix with PHiX Fast Camera
==================================================
"""
import sys
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from textwrap import dedent

import numpy as np
from raysect.core.scenegraph import print_scenegraph
from raysect.optical import World
from raysect.optical.observer.sampler2d import FullFrameSampler2D

from cherab.phix.machine import load_pfc_mesh
from cherab.phix.observer import load_camera
from cherab.phix.plasma import load_equilibrium
from cherab.phix.tools.raytransfer import load_rtc
from cherab.tools.raytransfer import RayTransferPipeline2D

# %%
# Path deffinition
# ----------------
ROOT_PATH = Path(__file__).parent.parent
dt_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
SAVE_DIR = ROOT_PATH / "output" / "RTM" / f"{dt_now}"
SAVE_DIR.mkdir(parents=True)


# %%
# Define custum logger
# --------------------
class Logger(TextIOBase):
    def __init__(self, filename: Path):
        self.console = sys.stdout
        self.file = open(filename, "w")

    def write(self, message: str):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


# set logger
sys.stdout = Logger(SAVE_DIR / "log.txt")


# %%
# Create scene-graph
# ------------------

# generate scene world
world = World()

# import phix equilibrium
eq = load_equilibrium()

# import phix machine configuration
mesh = load_pfc_mesh(world, reflection=True)

# import phix raytrasfer matrix
rtm = load_rtc(world, equilibrium=eq)

# import phix camera
camera = load_camera(world)

# %%
# Define Observer pipeline
# ------------------------
rtp = RayTransferPipeline2D(name="rtm")

# set camera's pipeline property
camera.pipelines = [rtp]

# %%
# Set camera parameters
# ---------------------
# calculate rtm for only (w, h) = (64, 192) element
# focus_pixel = (64, 192)
# mask = np.zeros(pixels, dtype=np.bool)
# mask[focus_pixel[0] - 1, focus_pixel[1] - 1] = True
camera.frame_sampler = FullFrameSampler2D()
camera.min_wavelength = 655.6
camera.max_wavelength = 656.8
camera.spectral_rays = 1
camera.spectral_bins = rtm.bins
camera.per_pixel_samples = 10
camera.lens_samples = 20

# %%
# Execute Ray-tracing
# ------------------
camera.observe()

# %%
# Save results
# ------------

# save config info as text
result = dedent(
    f"""
    ----------------------------------------------------------------------------------------
    camera name              : {camera.name}
    camera pixels            : {camera.pixels}
    camera per pixel samples : {camera.per_pixel_samples}
    camera lens samples      : {camera.lens_samples}
    camera pixel samples     : {camera.pixel_samples}
    camera spectral bins     : {camera.spectral_bins}
    camera wavelength range  : {camera.min_wavelength}, {camera.max_wavelength}
    ----------------------------------------------------------------------------------------
    PHiX PFCs                : {mesh}
    ----------------------------------------------------------------------------------------
    """
)

print(result)
print_scenegraph(world)

# save matrix
np.save(SAVE_DIR / "rtm.npy", rtp.matrix)

print(f"successfully saved in {SAVE_DIR}")
