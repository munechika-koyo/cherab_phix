"""Module to offer colour functionalities
"""
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import linspace, loadtxt, zeros

cimport cython
from numpy cimport float64_t, import_array, ndarray
from raysect.core.math.cython cimport clamp
from raysect.core.math.function.float.function1d.interpolate cimport Interpolator1DArray
from raysect.optical.colour cimport srgb_transfer_function
from raysect.optical.spectrum cimport Spectrum

ctypedef float64_t DTYPE_t

__all__ = [
    "resample_phantom_rgb",
    "spectrum_to_phantom_rgb",
    "phantom_rgb_to_srgb",
    "plot_samples",
    "plot_RGB_filter",
]

import_array()

# Path to directry saving fas-camera's RGB color filter curves
DIR = Path(__file__).parent / "sensitivity"

# load RGB sensitivity curv samples (unit [A/W])
R_samples = loadtxt(DIR / "R.txt", delimiter=",")
G_samples = loadtxt(DIR / "G.txt", delimiter=",")
B_samples = loadtxt(DIR / "B.txt", delimiter=",")

# interpolation using cubic spline
filter_r = Interpolator1DArray(R_samples[:, 0], R_samples[:, 1], "cubic", "nearest", 50.0)
filter_g = Interpolator1DArray(G_samples[:, 0], G_samples[:, 1], "cubic", "nearest", 50.0)
filter_b = Interpolator1DArray(B_samples[:, 0], B_samples[:, 1], "cubic", "nearest", 50.0)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] resample_phantom_rgb(double min_wavelength, double max_wavelength, int bins):
    """
    Pre-calculates samples of Phantom Camera's RGB sensitivity curves [A/W] over desired spectral range.

    Returns ndarray of shape [N, 3] where the last dimension (0, 1, 2) corresponds
    to (R, G, B).

    :param float min_wavelength: Lower wavelength bound on spectrum
    :param float max_wavelength: Upper wavelength bound on spectrum
    :param int bins: Number of spectral bins in spectrum
    :rtype: memoryview
    """

    cdef:
        int i
        double wavelength
        ndarray[DTYPE_t, ndim=2] rgb
        double[:, ::1] rgb_mv
        Spectrum spectrum

    if bins < 1:
        raise("Number of samples can not be less than 1.")

    if min_wavelength <= 0.0 or max_wavelength <= 0.0:
        raise ValueError("Wavelength can not be less than or equal to zero.")

    if min_wavelength >= max_wavelength:
        raise ValueError("Minimum wavelength can not be greater or equal to the maximum wavelength.")

    # Using Spectrum object to generate wavelengths
    spectrum = Spectrum(min_wavelength, max_wavelength, bins)
    rgb = zeros((spectrum.bins, 3))
    # memory view
    rgb_mv = rgb
    # sampling rgb filter in wavelengths
    for i, wavelength in enumerate(spectrum.wavelengths):
        rgb_mv[i, 0] = filter_r(wavelength)
        rgb_mv[i, 1] = filter_g(wavelength)
        rgb_mv[i, 2] = filter_b(wavelength)

    return rgb_mv


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef (double, double, double) spectrum_to_phantom_rgb(
    Spectrum spectrum,
    double[:, ::1] resampled_rgb=None,
    double exposure_time=1.0,
    double pixel_area=1.0
):
    """
    Calculate a tuple of R, G, B values from an input spectrum
    based on Phantom Hight-speed camera.
    The conversion equation from Spectral Power :math:`P(\\lambda)` [W/nm] to degital number DN [12bit]
    is represented as follows:

    .. math::

        DN = \\frac{t}{6.15\\times 10^{-9} A_{1\\text{px}}}\\int_{\\mathbb{R}} \\mathrm{d}\\lambda\\; S(\\lambda)P(\\lambda)

    where,

    - :math:`A_{1\\text{px}}` : 1 pixel area (in m$^2$)
    - :math:`S(\\lambda)` : spectral response at a specific wavelength (in A/W)
    - :math:`t` : exposure time (in second)
    - :math:`DN` : sensor response (in digital number, 12bits, from 0-4095)

    Parameters
    ----------
    spectrum : Spectrum
        raysect spectrum object (in radiance)
    resampled_rgb: memoryview
        Pre-calculated RGB sensitivity curves optimised
      for this spectral range, by default None
    exposure_time : float, optional
        exposure time of camera shutter (in [sec]), by default 1.0
    pixel_area : float, optional
        pixel area capturing light (in m$^2$), by default 1.0

    Returns
    -------
    tuple[float, float, float]
        R, G, B degital value in 12bit including over 4095 value
    """
    cdef:
        int index, bins
        double r = 0.0
        double g = 0.0
        double b = 0.0

    bins = spectrum.bins

    if resampled_rgb is None:
        resampled_rgb = resample_phantom_rgb(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

    for index in range(bins):
        r += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 0]
        g += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 1]
        b += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 2]

    r *= exposure_time / (6.15e-9 * pixel_area)
    g *= exposure_time / (6.15e-9 * pixel_area)
    b *= exposure_time / (6.15e-9 * pixel_area)

    return (r, g, b)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef (double, double, double) phantom_rgb_to_srgb(double r, double g, double b):
    """
    Convert Phantom Camera's RGB to sRGB in range [0, 1]

    phntom r, g, b in range [0, 4095]
    sr, sg, sb in range [0, 1]

    Parameters
    ----------
    r: float
        phantom R
    g: float
        phantom G
    b: float
        phantom B

    Returns
    -------
    tuple[float, float, float]
    """
    cdef:
        double sr, sg, sb

    # normalized
    sr = r / 4095
    sg = g / 4095
    sb = b / 4095

    # apply sRGB transfer function (gamma correction)
    sr = srgb_transfer_function(sr)
    sg = srgb_transfer_function(sg)
    sb = srgb_transfer_function(sb)

    # restrict to [0, 1]
    sr = clamp(sr, 0, 1)
    sg = clamp(sg, 0, 1)
    sb = clamp(sb, 0, 1)

    return (sr, sg, sb)


# plot
def plot_samples():
    """Plot RGB raw sensitivity curves of Phantom LAB110 camera.

    Example
    -------

    .. prompt:: python >>> auto

        >>> plot_samples()

    .. image:: ../_static/images/plots/rgb_sensitivity.png
    """
    fig, ax = plt.subplots()
    ax.plot(R_samples[:, 0], R_samples[:, 1], color="r", label="R")
    ax.plot(G_samples[:, 0], G_samples[:, 1], color="g", label="G")
    ax.plot(B_samples[:, 0], B_samples[:, 1], color="b", label="B")
    ax.set_xlim(350, 780)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("sensitivity [A/W]")
    plt.show()


def plot_RGB_filter(
    wavelengths=None, fig: Figure | None = None, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot interpolated RGB sensitivity curves of Phantom LAB 110 camera.

    This plot handles 1-D interpolated sensitivity data which is used to filter
    the light through the pipeline.

    Parameters
    ----------
    wavelengths : 1D vector-like, optional
        sampling points of wavelength, by default 500 points in range of (380, 780) [nm]
    fig
        matplotlib figure object
    ax
        matplotlib axes object

    Returns
    -------
    tuple[Figure, Axes]
        matplotlib figure and axes object

    Example
    -------

    .. prompt:: python >>> auto

        >>> import numpy as np
        >>> wavelengths = np.linspace(400, 600)
        >>> fig, ax = plot_RGB_filter(wavelengths)
        >>> fig.show()

    .. image:: ../_static/images/plots/rgb_filter.png
    """
    if wavelengths is None:
        wavelengths = linspace(350, 780, 500)

    if not isinstance(ax, Axes):
        if not isinstance(fig, Figure):
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            ax = fig.add_subplot()
    else:
        fig = ax.get_figure()

    ax.plot(wavelengths, [filter_r(i) for i in wavelengths], color="r", label="R")
    ax.plot(wavelengths, [filter_g(i) for i in wavelengths], color="g", label="G")
    ax.plot(wavelengths, [filter_b(i) for i in wavelengths], color="b", label="B")

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("sensitivity [A/W]")

    return (fig, ax)
