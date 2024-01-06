"""
Ray-tracing simulation of fast camera
=====================================
"""
import sys
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from textwrap import dedent

import numpy as np
from matplotlib import pyplot as plt
from raysect.optical import World
from raysect.optical.observer import PowerPipeline2D, RGBAdaptiveSampler2D, RGBPipeline2D

from cherab.phix.machine import load_pfc_mesh
from cherab.phix.observer import load_camera
from cherab.phix.plasma import load_plasma

# %%
# Path deffinition
# ----------------
ROOT = Path(__file__).parent.parent
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
SAVE_DIR = ROOT / "output" / "synthetic_images" / f"{time_now}"
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

# scene world
world = World()

# import plasma
plasma, eq = load_plasma(world)

# import phix mesh
mesh = load_pfc_mesh(world, reflection=True)

# import phix camera
camera = load_camera(world)

# %%
# Define Observer pipeline
# ------------------------
rgb = RGBPipeline2D(display_unsaturated_fraction=1.0, name="sRGB", display_progress=False)
power = PowerPipeline2D(display_progress=False, name="power")
camera.pipelines = [rgb, power]

# %%
# Set camera parameters
# ---------------------
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=10, cutoff=0.05)
camera.frame_sampler = sampler
camera.min_wavelength = 400
camera.max_wavelength = 780
camera.spectral_rays = 1
camera.spectral_bins = 20
camera.per_pixel_samples = 10
camera.lens_samples = 10

# %%
# Excute Ray-tracing
# ------------------
plt.ion()
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
)[1:-1]
(SAVE_DIR / "results.txt").write_text(result)
print(result)

# save rgb image
rgb.save(str(SAVE_DIR / "rgb.png"))

# save power data
power.save(str(SAVE_DIR / "power.png"))
np.save(SAVE_DIR / "power.npy", power.frame.mean)

print(f"successfully saved in {SAVE_DIR}.")
