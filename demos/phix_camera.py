import numpy as np
from matplotlib import pyplot as plt
import datetime

from raysect.core.math import AffineMatrix3D, translate, rotate_z
from raysect.optical import World
from raysect.optical.observer import RGBPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D
from cherab.phix.plasma import import_plasma
from cherab.phix.machine import import_phix_mesh
from cherab.phix.observer.camera import LensCamera
from cherab.phix.machine.material import SUS316L

# generate scene world
world = World()

# import plasma
plasma, eq = import_plasma(world)

# import phix mesh
mesh = import_phix_mesh(world, vessel_material=SUS316L(), reflection=True)

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
rgb = RGBPipeline2D(display_unsaturated_fraction=0.98, name="sRGB")
pipelines = [rgb]
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=10, cutoff=0.05)

# camera = PinholeCamera(
#     (256, 512), parent=world, pipelines=pipelines, transform=camera_trans * camera_rot
# )
pixsel = (128, 256)
camera = LensCamera(
    pixels=pixsel,
    width=25.6e-3 * 256 / 1280,
    focal_length=1.0e-2,
    working_distance=50.0e-2,
    F_value=3 * (22 - 3.5) / 10 + 3.5,
    parent=world,
    pipelines=pipelines,
    transform=camera_trans * camera_rot,
)
# camera.frame_sampler = sampler
camera.min_wavelength = 400
camera.max_wavelength = 780
camera.spectral_rays = 1
camera.spectral_bins = 20
camera.pixel_samples = 5

plt.ion()
camera.observe()

rgb.save(f"../output/test_camera({datetime.datetime.now()}).png")
plt.ioff()
# plt.show()
