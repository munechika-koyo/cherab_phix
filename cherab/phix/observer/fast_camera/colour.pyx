# cython: language_level=3

import os
from matplotlib import pyplot as plt
from numpy import loadtxt, zeros, linspace
from numpy cimport ndarray, float64_t
from raysect.core.math.cython cimport clamp
from raysect.core.math.function.float.function1d.interpolate cimport Interpolator1DArray
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.colour cimport srgb_transfer_function
cimport cython

ctypedef float64_t DTYPE_t

# Path to directry saving fas-camera's RGB color filter curves
DIR = os.path.join(os.path.dirname(__file__), "sensitivity")

# load RGB sensitivity curv samples (unit [A/W])
R_samples = loadtxt(os.path.join(DIR, "R.txt"), delimiter=",")
G_samples = loadtxt(os.path.join(DIR, "G.txt"), delimiter=",")
B_samples = loadtxt(os.path.join(DIR, "B.txt"), delimiter=",")

# interpolation using cubic spline
filter_r = Interpolator1DArray(R_samples[:, 0], R_samples[:, 1], "cubic", "nearest", 50.0)
filter_g = Interpolator1DArray(G_samples[:, 0], G_samples[:, 1], "cubic", "nearest", 50.0)
filter_b = Interpolator1DArray(B_samples[:, 0], B_samples[:, 1], "cubic", "nearest", 50.0)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] resample_phantom_rgb(double min_wavelength, double max_wavelength, int bins):
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
cpdef (double, double, double) spectrum_to_phantom_rgb(Spectrum spectrum, double[:,::1] resampled_rgb=None, double exposure_time=1.0, double pixel_area=1.0):
    """
    Calculates a tuple of R, G, B values from an input spectrum
    based on Phantom Hight-speed camera.
    The conversion equation from Spectral Power [W/nm] to degital value [12bit]
    is represented as follows:

    .. math::
        W [W/nm] = 6.15\\times 10^{-9} \\cdot \\frac{A_\\text{1px}}{S(\\lambda)\\cdot t}\\cdot DN,

    where,
    - $A_${1px} : 1 pixel area (in m$^2$)
    - $S(\\lambda)$ : spectral response at a specific wavelength (in A/W)
    - $t$ : exposure time (in second)
    - DN : sensor response (in digital number, 12bits, from 0-4095)

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
    tuple
        R, G, B degital value in 12bit including over 4095 value
    """
    cdef:
        int index, bins
        double r, g, b

    bins = spectrum.bins

    if resampled_rgb is None:
        resampled_rgb = resample_phantom_rgb(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

    r = 0
    g = 0
    b = 0
    for index in range(bins):
        r += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 0]
        g += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 1]
        b += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_rgb[index, 2]

    r *= exposure_time / (6.15e-9 * pixel_area)
    g *= exposure_time / (6.15e-9 * pixel_area)
    b *= exposure_time / (6.15e-9 * pixel_area)

    return r, g, b

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
    tuple
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

    return sr, sg, sb


# plot
def plot_samples():
    """Plot RGB sensitivity curvs samplong points of Phantom camera.
    """
    fig, ax = plt.subplots()
    ax.plot(R_samples[:, 0], R_samples[:, 1], color="r", label="R")
    ax.plot(G_samples[:, 0], G_samples[:, 1], color="g", label="G")
    ax.plot(B_samples[:, 0], B_samples[:, 1], color="b", label="B")
    ax.set_xlim(350, 780)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("sensitivity [A/W]")
    plt.show()


def plot_RGB_filter(wavelengths=None):
    """Plot RGB sensitivity curvs of Phantom camera.

    Parameters
    ----------
    wavelengths : 1D vector-like, optional
        sampling points of wavelength, by default 500 points in range of (380, 780) [nm]
    """
    if wavelengths is None:
        wavelengths = linspace(350, 780, 500)

    fig, ax = plt.subplots()
    ax.plot(wavelengths, [filter_r(i) for i in wavelengths], color="r", label="R")
    ax.plot(wavelengths, [filter_g(i) for i in wavelengths], color="g", label="G")
    ax.plot(wavelengths, [filter_b(i) for i in wavelengths], color="b", label="B")

    ax.set_ylim(0, 0.180)
    ax.set_xlim(350, 780)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("sensitivity [A/W]")
    plt.show()