"""
Ray-tracing simulation of fast camera
=====================================
Here, we simulate fast camera measurement with different pipelines and
focus on three hydrogen balmer lines emission.
"""
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
from raysect.optical import World
from raysect.optical.observer import (
    FullFrameSampler2D,
    RGBAdaptiveSampler2D,
    SpectralPowerPipeline2D,
)

from cherab.phix.machine import import_phix_mesh
from cherab.phix.observer import import_phix_camera
from cherab.phix.observer.fast_camera.colour import RGBPipeline2D
from cherab.phix.plasma import import_plasma

# Source root path
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
    display_unsaturated=True, name="PhantomRGB", display_progress=False, exposure_time=99e-6
)
sp = SpectralPowerPipeline2D(name="Spectrul_Power")
camera.pipelines = [rgb, sp]

# %%
# Define Observer sampler
# ------------------------
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=10, cutoff=0.05)
# sampler = FullFrameSampler2D()
camera.frame_sampler = sampler

# %%
# Set camera parameters
# ---------------------
# pixels = (128, 256)  # [px]
# camera.pixels = pixels  # if you change pixel resolution
camera.spectral_rays = 1
camera.spectral_bins = 50
camera.per_pixel_samples = 10
camera.lens_samples = 10

# iteration for three wavelength range
Halpha = (655.6, 656.8)
Hbeta = (485.6, 486.5)
Hgamma = (433.6, 434.4)

# %%
# Excute Ray-tracing
# ------------------
xyz_frames = []
sp_frames_mean = []
wavelengths = []
for i, rage_wavelength in enumerate([Halpha, Hbeta, Hgamma]):

    camera.min_wavelength = rage_wavelength[0]
    camera.max_wavelength = rage_wavelength[1]

    # calculate ray-tracing
    camera.observe()

    # store piplines
    xyz_frames.append(rgb.xyz_frame.copy())
    sp_frames_mean.append(sp.frame.mean.copy())
    wavelengths.append(sp.wavelengths.copy())

xyz_frame = xyz_frames[0].copy()

for x in range(rgb.xyz_frame.nx):
    for y in range(rgb.xyz_frame.ny):
        for xyz_frame_mv in [xyz_frames[1], xyz_frames[2]]:
            xyz_frame.combine_samples(
                x,
                y,
                0,
                xyz_frame_mv.mean[x, y, 0],
                xyz_frame_mv.variance[x, y, 0],
                camera.pixel_samples,
            )
            xyz_frame.combine_samples(
                x,
                y,
                1,
                xyz_frame_mv.mean[x, y, 1],
                xyz_frame_mv.variance[x, y, 1],
                camera.pixel_samples,
            )
            xyz_frame.combine_samples(
                x,
                y,
                2,
                xyz_frame_mv.mean[x, y, 2],
                xyz_frame_mv.variance[x, y, 2],
                camera.pixel_samples,
            )

# create sRGB image
srgb_img = rgb._generate_display_image(xyz_frame)

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
    display_sensitivity      : {rgb.display_sensitivity}
    ----------------------------------------------------------------------------------------
    PHiX PFCs                : {mesh}
    ----------------------------------------------------------------------------------------
    """
)[1:-1]
(save_dir / "results.txt").write_text(result)
print(result)

# save xyz_frame, srg_image
np.save(save_dir / "xyz_frame_mean.npy", xyz_frame.mean)
np.save(save_dir / "srgb_img.npy", srgb_img)

# save wavelength
np.save(save_dir / "wavelengths_Halpha.npy", wavelengths[0])
np.save(save_dir / "wavelengths_Hbeta.npy", wavelengths[1])
np.save(save_dir / "wavelengths_Hgamma.npy", wavelengths[2])

# # save spectral power mean
np.save(save_dir / "SP_frame_mean_Halpha.npy", sp_frames_mean[0])
np.save(save_dir / "SP_frame_mean_Hbeta.npy", sp_frames_mean[1])
np.save(save_dir / "SP_frame_mean_Hgamma.npy", sp_frames_mean[2])

print(f"successfully saved in {save_dir}.")
