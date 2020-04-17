import numpy as np
from matplotlib import pyplot as plt
import datetime

from raysect.core.math import AffineMatrix3D, translate, rotate_z
from raysect.optical import World
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, PowerPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D, MonoAdaptiveSampler2D
from cherab.tools.raytransfer import RayTransferPipeline2D
from cherab.phix.plasma import import_plasma
from cherab.phix.machine import import_phix_mesh
from cherab.phix.tools import import_phix_rtm

# generate scene world
world = World()

# import plasma
plasma, eq = import_plasma(world)

# import phix mesh
mesh = import_phix_mesh(world, reflection=True)

# set Ray Transfer Matrix
rtm = import_phix_rtm(world, equilibrium=eq, grid_size=2.0e-3)

# creating ray transfer pipeline
rtp = RayTransferPipeline2D()

# calculate ray-tracing
plt.ion()

# camera
# import camera extinct paras
translation_vector = np.loadtxt("translation_vector.csv")
rotation_mat = np.loadtxt("rotation_matrix.csv")

orientation = rotation_mat.T
camera_pos = -np.dot(orientation, translation_vector.reshape((3, 1)))
camera_trans = translate(*camera_pos)
camera_rot = np.block([[orientation, np.zeros((3, 1))], [np.array([0, 0, 0, 1])]])
camera_rot = AffineMatrix3D(camera_rot) * rotate_z(180)

# set pipelines and sampler
rgb = RGBPipeline2D(display_unsaturated_fraction=1.0, name="sRGB")
power = PowerPipeline2D(display_unsaturated_fraction=1.0, name="Power")
rgb_sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=100, cutoff=0.05)
mono_sampler = MonoAdaptiveSampler2D(power, min_samples=100)
# pipelines = [rgb, power]
pipelines = [rtp]

pixsel = (128, 256)  # [px]
# pixsel = (256, 512)  # [px]
pixel_pitch = 20e-6  # [m]
focal_length = 10e-3
fov = 2 * np.arctan(0.5 * 256 * pixel_pitch / focal_length) * 180 / np.pi
camera = PinholeCamera(
    pixsel, fov=fov * 2, parent=world, pipelines=pipelines, transform=camera_trans * camera_rot
)

# camera.frame_sampler = mono_sampler
camera.min_wavelength = 600
camera.max_wavelength = 601
camera.spectral_rays = 1
camera.spectral_bins = rtm.bins
camera.pixel_samples = 10

plt.ion()
camera.observe()

filename = f"../output/test_pinhole_camera_rtm({datetime.datetime.now()})"
# rgb.save(filename + ".png")
# np.save(filename, power.frame.mean)
# power.save(filename + "_power")
np.save(filename, rtp.matrix)
plt.ioff()
# plt.show()
