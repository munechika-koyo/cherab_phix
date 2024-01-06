from contextlib import nullcontext as does_not_raise
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid
from raysect.optical import World

from cherab.phix.tools.raytransfer import import_phix_rtc
from cherab.phix.tools.visualize import (
    set_axis_properties,
    set_cbar_format,
    set_norm,
    show_profile,
    show_profiles,
)


@pytest.fixture
def axes():
    return plt.subplots(1, 1)[1]


@pytest.fixture
def profile2d():
    path = Path(__file__).parent / "data" / "Halpha_emissivity.npz"
    return np.load(path)["emiss"]


def test_set_axis_properties(axes):
    set_axis_properties(axes)


@pytest.mark.parametrize(
    ["formatter", "kwargs", "expected_formatter", "expectation"],
    [
        pytest.param("linear", {}, "scalar", does_not_raise(), id="linear"),
        pytest.param("scalar", {}, "scalar", does_not_raise(), id="scalar"),
        pytest.param("log", {}, "log", does_not_raise(), id="log"),
        pytest.param("symlog", {"linear_width": 1.0}, "log", does_not_raise(), id="symlog"),
        pytest.param("asinh", {}, "asinh", pytest.raises(NotImplementedError), id="asinh"),
        pytest.param("percent", {}, "percent", does_not_raise(), id="percent"),
        pytest.param("eng", {"unit": "m"}, "eng", does_not_raise(), id="eng"),
    ],
)
def test_set_cbar_format(formatter, kwargs, expected_formatter, expectation):
    _, ax = plt.subplots()
    with expectation:
        set_cbar_format(ax, formatter, **kwargs)
        assert expected_formatter.capitalize() in ax.yaxis.get_major_formatter().__class__.__name__


@pytest.mark.parametrize(
    ["mode", "vmin", "vmax", "expected_norm", "expectation"],
    [
        pytest.param("scalar", 0.0, 1.0, "Normalize", does_not_raise(), id="scalar"),
        pytest.param("linear", 0.0, 1.0, "Normalize", does_not_raise(), id="linear"),
        pytest.param("log", 1e-3, 1.0, "LogNorm", does_not_raise(), id="log"),
        pytest.param("log", -1.0, 1.0, "LogNorm", pytest.raises(ValueError), id="log_negative"),
        pytest.param("symlog", -1.0, 1.0, "SymLogNorm", does_not_raise(), id="symlog (-1, 1)"),
        pytest.param("symlog", 1.0, -1.0, "SymLogNorm", does_not_raise(), id="symlog (1, -1)"),
        pytest.param(
            "asinh", 0.0, 1.0, "AsinhNorm", pytest.raises(NotImplementedError), id="asinh"
        ),
    ],
)
def test_set_norm(mode, vmin, vmax, expected_norm, expectation):
    with expectation:
        norm = set_norm(mode, vmin, vmax, linear_width=1.0)
        assert expected_norm in norm.__class__.__name__


@pytest.mark.parametrize(
    ["rtc", "vmax", "vmin", "plot_contour", "levels", "plot_mode", "expected_num_lines"],
    [
        pytest.param(None, None, None, False, None, "scalar", None, id="no contour"),
        pytest.param(None, None, None, True, None, "scalar", 6, id="default (contour)"),
        pytest.param(
            None, None, None, True, np.linspace(0, 12e3, 10), "scalar", 8, id="custom levels"
        ),
        pytest.param(
            import_phix_rtc(World()), None, None, False, None, "scalar", None, id="no contour (rtc)"
        ),
    ],
)
def test_show_profile(
    rtc, vmax, vmin, plot_contour, levels, plot_mode, expected_num_lines, axes, profile2d
):
    lines = show_profile(
        axes,
        profile2d,
        rtc=rtc,
        vmax=vmax,
        vmin=vmin,
        plot_contour=plot_contour,
        levels=levels,
        plot_mode=plot_mode,
    )
    if expected_num_lines is not None:
        assert isinstance(lines, list)
        assert len(lines) == expected_num_lines
    else:
        assert lines is None

    fig = axes.get_figure()
    fig.show()
    plt.close(fig=fig)


@pytest.mark.parametrize(
    ["num_profiles", "fig", "nrow_ncols", "rtc", "cbar_mode", "plot_mode", "expectation"],
    [
        pytest.param(1, None, None, None, "single", "scalar", does_not_raise(), id="1 profile"),
        pytest.param(
            -1, None, None, None, "single", "scalar", pytest.raises(TypeError), id="invalid profile"
        ),
        pytest.param(2, None, None, None, "single", "scalar", does_not_raise(), id="2 profiles"),
        pytest.param(4, None, None, None, "single", "scalar", does_not_raise(), id="4 profiles"),
        pytest.param(
            4,
            None,
            (2, 2),
            None,
            "single",
            "scalar",
            does_not_raise(),
            id="4 profiles (nrow_ncols)",
        ),
        pytest.param(
            1,
            None,
            None,
            import_phix_rtc(World()),
            "single",
            "scalar",
            does_not_raise(),
            id="1 profile (rtc)",
        ),
        pytest.param(
            2,
            None,
            None,
            None,
            "each",
            "scalar",
            does_not_raise(),
            id="2 profiles (each color bars)",
        ),
    ],
)
def test_show_profiles(
    num_profiles, fig, nrow_ncols, rtc, cbar_mode, plot_mode, expectation, profile2d
):
    if num_profiles == 1:
        profiles = profile2d
    elif num_profiles < 0:
        profiles = "invalid profiles"
    else:
        profiles = [profile2d] * num_profiles

    with expectation:
        fig, grids = show_profiles(
            profiles,
            fig=fig,
            nrow_ncols=nrow_ncols,
            rtc=rtc,
            cbar_mode=cbar_mode,
            plot_mode=plot_mode,
        )

        assert isinstance(fig, Figure)
        assert isinstance(grids, ImageGrid)
        assert len(grids) == num_profiles

        nrow, ncol = grids.get_geometry()
        if nrow_ncols is None:
            assert nrow == 1
            assert ncol == num_profiles
        else:
            assert nrow == nrow_ncols[0]
            assert ncol == nrow_ncols[1]

        assert cbar_mode == grids._colorbar_mode

        fig.show()
        plt.close(fig=fig)
