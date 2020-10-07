import os
import numpy as np
import datetime

from raysect.optical import World
from raysect.optical.observer import PowerPipeline2D, SpectralPowerPipeline2D
from cherab.phix.observer.fast_camera.colour import RGBPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D
from cherab.phix.plasma import import_plasma
from cherab.phix.machine import import_phix_mesh
from cherab.phix.observer import import_phix_camera
from raysect.optical.observer import FullFrameSampler2D


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
rgb = RGBPipeline2D(display_unsaturated=True, name="PhantomRGB", display_progress=False, exposure_time=99e-6)
sp = SpectralPowerPipeline2D(name="Spectrul_Power")
camera.pipelines = [rgb, sp]
# sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=10, cutoff=0.05)

# camera.frame_sampler = sampler
# pixels = (256, 512)  # [px]
pixels = (128, 256)  # [px]

# focus_pixel = (128, 256)
# mask = np.zeros(pixels, dtype=np.bool)
# mask[focus_pixel[0] - 1, focus_pixel[1] - 1] = True
# camera.frame_sampler = FullFrameSampler2D(mask)
camera.frame_sampler = FullFrameSampler2D()
camera.pixels = pixels
camera.spectral_rays = 1
camera.spectral_bins = 50
camera.per_pixel_samples = 5
camera.lens_samples = 10
# iteration of three wavelength range
Halpha = (655.6, 656.8)
Hbeta = (485.6, 486.5)
Hgamma = (433.6, 434.4)
# H_alpha_peak = 656.0
# H_beta_peak = 486.0
# H_gamma_peak = 434.0

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
                x, y, 0, xyz_frame_mv.mean[x, y, 0], xyz_frame_mv.variance[x, y, 0], camera.pixel_samples
            )
            xyz_frame.combine_samples(
                x, y, 1, xyz_frame_mv.mean[x, y, 1], xyz_frame_mv.variance[x, y, 1], camera.pixel_samples
            )
            xyz_frame.combine_samples(
                x, y, 2, xyz_frame_mv.mean[x, y, 2], xyz_frame_mv.variance[x, y, 2], camera.pixel_samples
            )

srgb_img = rgb._generate_display_image(xyz_frame)
# save
time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = os.path.join(DIR, "output", f"{time_now}")
os.makedirs(save_dir)

# save xyz_frame, srg_image
np.save(os.path.join(save_dir, "xyz_frame_mean"), xyz_frame.mean)
np.save(os.path.join(save_dir, "srgb_img"), srgb_img)

# save wavelength
np.save(os.path.join(save_dir, "wavelengths_Halpha"), wavelengths[0])
np.save(os.path.join(save_dir, "wavelengths_Hbeta"), wavelengths[1])
np.save(os.path.join(save_dir, "wavelengths_Hgamma"), wavelengths[2])

# # save spectral power mean
np.save(os.path.join(save_dir, "SP_frame_mean_Halpha"), sp_frames_mean[0])
np.save(os.path.join(save_dir, "SP_frame_mean_Hbeta"), sp_frames_mean[1])
np.save(os.path.join(save_dir, "SP_frame_mean_Hgamma"), sp_frames_mean[2])

# record infromation
with open(os.path.join(save_dir, "result.txt"), "w") as f:
    result = (
        "--------------------------------------------------------------------------------\n"
        f"display_sensitivity      : {rgb.display_sensitivity}\n"
        f"camera pixels            : {camera.pixels}\n"
        f"camera per pixel samples : {camera.per_pixel_samples}\n"
        f"camera lens samples      : {camera.lens_samples}\n"
        f"camera pixel samples     : {camera.pixel_samples}\n"
        f"camera spectral bins     : {camera.spectral_bins}\n"
        "--------------------------------------------------------------------------------\n"
    )
    f.write(result)


# rgb.save(filename + ".png")
# # np.save(filename + "_power", power.frame.mean)
# print(f"successfully saved {filename.split(os.path.sep)[-1]}")
