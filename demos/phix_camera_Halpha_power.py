"""
Ray-tracing simulation of fast camera
=====================================
Here, we simulate fast camera measurement focusing on H-alpha emission.
"""
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
from matplotlib import pyplot as plt
from raysect.optical import World
from raysect.optical.observer import PowerPipeline2D, SpectralPowerPipeline2D

from cherab.phix.machine import load_pfc_mesh
from cherab.phix.observer import load_camera
from cherab.phix.plasma import load_plasma

ROOT = Path(__file__).parent.parent

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
power = PowerPipeline2D(name="power")
spectral = SpectralPowerPipeline2D(name="Spectrul_Power")
camera.pipelines = [power, spectral]

# %%
# Set camera parameters
# ---------------------
camera.pixels = (128, 256)
camera.min_wavelength = 655.5
camera.max_wavelength = 657
camera.spectral_rays = 1
camera.spectral_bins = 50
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
power.save(str(save_dir / "power.png"))
np.save(save_dir / "power.npy", power.frame.mean)
np.save(save_dir / "spectrum.npy", spectral.frame.mean)
np.save(save_dir / "wavelength.npy", spectral.wavelengths)
print(f"successfully saved in {save_dir}.")
