# import numpy as np
from matplotlib import pyplot as plt

from raysect.optical import World, Ray, Point3D, Vector3D
from cherab.phix.plasma import import_plasma

# generate scene world
world = World()

# import plasma
plasma = import_plasma(world)

# terminal absorption


# calculate ray-tracing
# plt.ion()

r = Ray(
    origin=Point3D(-1, 0, 0),
    direction=Vector3D(1, 0, 0),
    min_wavelength=100,
    max_wavelength=1000,
    bins=1e6,
)
s = r.trace(world)
plt.plot(s.wavelengths, s.samples)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance (W/m^2/str/nm)")
plt.title("Sampled Spectrum")
plt.show()
