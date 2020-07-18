import os
import numpy as np
from matplotlib import pyplot as plt
from raysect.optical import InterpolatedSF
from cherab.core.math.interpolators import Interpolate1DCubic


# Path to directry saving fas-camera's RGB color filter curves
DIR = os.path.join(os.path.dirname(__file__), "sensitivity")

# load RGB sensitivity curv samples (unit [A/W])
R_samples = np.loadtxt(os.path.join(DIR, "R.txt"), delimiter=",")
G_samples = np.loadtxt(os.path.join(DIR, "G.txt"), delimiter=",")
B_samples = np.loadtxt(os.path.join(DIR, "B.txt"), delimiter=",")

# interpolation using cubic spline
filter_r = Interpolate1DCubic(R_samples[:, 0], R_samples[:, 1], extrapolate=True, extrapolation_type="nearest")
filter_g = Interpolate1DCubic(G_samples[:, 0], G_samples[:, 1], extrapolate=True, extrapolation_type="nearest")
filter_b = Interpolate1DCubic(B_samples[:, 0], B_samples[:, 1], extrapolate=True, extrapolation_type="nearest")


def spectrum_to_rgb_ampere(spectrum):
    """
    Calculates a tuple of R, G, B values from an input spectrum
    If the spectrum unit is [W/nm] then it is converted to ampere [A]

    Parameters
    ----------
    spectrum : Spectrum
        raysect spectrum object

    Returns
    -------
    tuple
        R, G, B values [A]
    """

    resampled_rgb = np.zeros((spectrum.bins, 3))
    for j, wavelength in enumerate(spectrum.wavelengths):
        resampled_rgb[j, 0] = filter_r(wavelength)
        resampled_rgb[j, 1] = filter_g(wavelength)
        resampled_rgb[j, 2] = filter_b(wavelength)

    if resampled_rgb.shape[0] != spectrum.bins or resampled_rgb.shape[1] != 3:
        raise ValueError(
            "The supplied resampled_rgb array size is inconsistent with the number of spectral bins or channel count."
        )

    r = 0
    g = 0
    b = 0
    for index in range(spectrum.bins):
        r += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 0]
        g += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 1]
        b += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 2]

    return r, g, b


# plot
def plot_samples():
    fig, ax = plt.subplots()
    ax.plot(R_samples[:, 0], R_samples[:, 1], color="r", label="R")
    ax.plot(G_samples[:, 0], G_samples[:, 1], color="g", label="G")
    ax.plot(B_samples[:, 0], B_samples[:, 1], color="b", label="B")
    ax.set_xlim(350, 780)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("sensitivity [A/W]")
    plt.show()


def plot_RGB_filter(wavelength=None):
    if wavelength is None:
        wavelengths = np.linspace(350, 780, 500)

    fig, ax = plt.subplots()
    ax.plot(wavelengths, [filter_r(i) for i in wavelengths], color="r", label="R")
    ax.plot(wavelengths, [filter_g(i) for i in wavelengths], color="g", label="G")
    ax.plot(wavelengths, [filter_b(i) for i in wavelengths], color="b", label="B")

    ax.set_ylim(0, 0.180)
    ax.set_xlim(350, 780)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("sensitivity [A/W]")
    plt.show()
