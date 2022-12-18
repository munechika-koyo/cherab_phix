"""
Ray-tracing simulation of fast camera
=====================================
"""
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
from matplotlib import pyplot as plt
from raysect.optical import World
from raysect.optical.observer import (
    PowerPipeline2D,
    RGBAdaptiveSampler2D,
    RGBPipeline2D,
)

from cherab.phix.machine import import_phix_mesh
from cherab.phix.observer import import_phix_camera
from cherab.phix.plasma import import_plasma

ROOT = Path(__file__).parent.parent

# %%
# Create scene-graph
# ------------------

# scene world
world = World()

# import plasma
plasma, eq = import_plasma(world)

# import phix mesh
mesh = import_phix_mesh(world, reflection=True)

# import phix camera
camera = import_phix_camera(world)

# %%
# Define Observer pipeline
# ------------------------
rgb = RGBPipeline2D(
    display_unsaturated_fraction=1.0, name="sRGB", display_progress=False
)
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
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = ROOT / "output" / "synthetic_images" / f"{time_now}"
save_dir.mkdir(parents=True)

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
(save_dir / "results.txt").write_text(result)
print(result)
rgb.save(str(save_dir / "rgb.png"))
power.save(str(save_dir / "power.png"))
np.save(save_dir / "power.npy", power.frame.mean)
print(f"successfully saved in {save_dir}.")
