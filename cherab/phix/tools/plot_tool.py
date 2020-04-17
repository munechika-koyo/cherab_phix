# import numpy as np
from matplotlib import pyplot as plt


def plot_ray(ray, world=None):
    f"""plotting the spectrum of one ray-tracing

    Parameters
    ----------
    ray : Ray
        raysect Ray object
    world : Node, optional
        raysect Node object, by default None
    """
    s = ray.trace(world)
    plt.figure()
    plt.plot(s.wavelengths, s.samples)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (W/m$^2$/str/nm)")
    plt.title("Sampled Spectrum")
    plt.show()
