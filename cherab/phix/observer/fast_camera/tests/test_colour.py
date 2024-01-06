import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from raysect.optical import Spectrum

from cherab.phix.observer.fast_camera.colour import (
    phantom_rgb_to_srgb,
    plot_RGB_filter,
    plot_samples,
    resample_phantom_rgb,
    spectrum_to_phantom_rgb,
)


@pytest.mark.parametrize(
    ["r", "g", "b", "expected"],
    [
        pytest.param(0.0, 0.0, 0.0, (0.0, 0.0, 0.0), id="black"),
        pytest.param(
            1.0,
            1.0,
            1.0,
            (0.003155067155067155, 0.003155067155067155, 0.003155067155067155),
            id="white",
        ),
        pytest.param(
            0.5,
            0.5,
            0.5,
            (0.0015775335775335775, 0.0015775335775335775, 0.0015775335775335775),
            id="grey",
        ),
    ],
)
def test_phantom_rgb_to_srgb(r, g, b, expected):
    assert phantom_rgb_to_srgb(r, g, b) == expected


@pytest.mark.parametrize(
    ["wavelengths", "figure", "axes"],
    [
        pytest.param(None, None, None, id="no args"),
        pytest.param([400, 405, 410], None, None, id="wavelengths"),
        pytest.param(None, plt.figure(), None, id="figure"),
        pytest.param(None, None, plt.axes(), id="axes"),
        pytest.param(None, plt.figure(), plt.axes(), id="figure and axes"),
    ],
)
def test_plot_RGB_filter(wavelengths, figure, axes):
    fig, ax = plot_RGB_filter(wavelengths, figure, axes)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_samples():
    plot_samples()


@pytest.mark.parametrize(
    ["min_wavelength", "max_wavelength", "bins"],
    [
        pytest.param(400, 700, 10, id="default"),
        pytest.param(400, 700, 20, id="20 bins"),
        pytest.param(400, 700, 30, id="30 bins"),
    ],
)
def test_resample_phantom_rgb(min_wavelength, max_wavelength, bins):
    rgb_mv = resample_phantom_rgb(min_wavelength, max_wavelength, bins)
    assert len(rgb_mv) == bins
    assert len(rgb_mv[0]) == 3


@pytest.mark.parametrize(
    ["spectrum", "exposure_time", "pixel_area"],
    [
        pytest.param(Spectrum(380, 720, 250), 1.0, 1.0, id="default"),
        pytest.param(Spectrum(655, 657, 50), 1.0, 1.0, id="Halpha"),
        pytest.param(Spectrum(380, 720, 250), 0.5, 1.0, id="0.5 exposure time"),
        pytest.param(Spectrum(380, 720, 250), 1.0, 0.5, id="0.5 pixel area"),
    ],
)
def test_spectrum_to_phantom_rgb(spectrum, exposure_time, pixel_area):
    rgb = spectrum_to_phantom_rgb(spectrum, exposure_time=exposure_time, pixel_area=pixel_area)
    assert len(rgb) == 3
