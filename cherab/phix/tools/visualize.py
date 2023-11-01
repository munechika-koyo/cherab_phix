"""Modules to offer visualization tools."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import (
    AutoLocator,
    AutoMinorLocator,
    EngFormatter,
    LogFormatterSciNotation,
    LogLocator,
    MultipleLocator,
    PercentFormatter,
    ScalarFormatter,
    SymmetricalLogLocator,
)
from mpl_toolkits.axes_grid1.axes_grid import CbarAxesBase, ImageGrid
from numpy.typing import NDArray
from raysect.optical import Ray, World

from cherab.tools.raytransfer.raytransfer import RayTransferCylinder

from ..machine.wall_outline import INNER_LIMITER, OUTER_LIMITER
from .raytransfer import import_phix_rtc
from .utils import calc_contours

__all__ = [
    "plot_ray",
    "show_phix_profiles",
    "show_phix_profile",
    "set_axis_properties",
    "set_cbar_format",
    "set_norm",
]


def plot_ray(ray: Ray, world: World):
    """Plotting the spectrum of one ray-tracing.

    Parameters
    ----------
    ray
        raysect Ray object
    world
        raysect Node object, by default None
    """
    s = ray.trace(world)
    plt.figure()
    plt.plot(s.wavelengths, s.samples)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (W/m$^2$/str/nm)")
    plt.title("Sampled Spectrum")
    plt.show()


def show_phix_profiles(
    profiles: list[NDArray] | NDArray,
    fig: Figure | None = None,
    nrow_ncols: tuple[int, int] | None = None,
    clabel: str = "",
    cmap: str = "inferno",
    rtc: RayTransferCylinder | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    axes_pad: float = 0.02,
    cbar_mode: str = "single",
    plot_mode: str = "scalar",
    linear_width: float = 1.0,
) -> tuple[Figure, ImageGrid]:
    """Show in-phix-limiter 2D profiles such as emission profile.

    This function can show several 2D profiles with matplotlib imshow style.

    Parameters
    ----------
    profiles
         2D-array-like (nr, nz) profile inner PHiX limiter
         if you want to show several profiles, you must put one list containing 2D array-like data.
    fig
        Figure object, by default `plt.figure()`
    nrow_ncols
        ImageGrid's parameter to set the number of rows and columns, by default None.
        If None, the number of rows and columns are set to ``(1, len(profiles))``
    clabel
        colobar label
    cmap
        color map, by default ``"inferno"``
    rtc
        cherab's raytransfer objects, by default the instance loaded by `.import_phix_rtc`.
    vmax
        to set the upper color limitation, by default maximum value of all profiles,
        if ``cbar_mode=="single"``
    vmin
        to set the lower color limitation, by default minimal value of all profiles,
        if ``cbar_mode=="single"``
    axes_pad
        ImageGrid's parameter to set the interval between axes, by default 0.02
    cbar_mode
        ImgeGrid's parameter to set colorbars in ``"single"`` axes or ``"each"`` axes,
        by default ``"single"``
    scientific_notation
        whether or not to set colorbar format with scientific notation, by default True
    plot_mode
        the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``, ``"symlog"``, ``"asinh"``},
        by default ``"scalar"``.
        Each mode corresponds to the :obj:`~matplotlib.colors.Normalize` object as follows.
    linear_width
        linear width of asinh/symlog norm, by default 1.0

    Returns
    -------
    tuple[:obj:`~matplotlib.figure.Figure`, :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`]
        tuple containing matplotlib figure object and instance of ImageGrid.
    """
    # transform the type of argument if it is not list type.
    if not isinstance(profiles, list):
        profiles = [profiles]

    # set ImageGrid
    if not isinstance(fig, Figure):
        fig = plt.figure()

    if nrow_ncols is None:
        nrow_ncols = (1, len(profiles))

    grids = ImageGrid(
        fig,
        111,
        nrows_ncols=nrow_ncols,
        axes_pad=axes_pad,
        cbar_mode=cbar_mode,
        cbar_pad=0.0,
    )

    # get maximum and minimum value of each profile
    _vmaxs = [profile.max() for profile in profiles]
    _vmins = [profile.min() for profile in profiles]

    # define vmaxs
    if isinstance(vmax, (float, int)):
        vmaxs: list[float] = [vmax for _ in range(len(profiles))]
    else:
        vmaxs: list[float] = _vmaxs

    # define vmins
    if isinstance(vmin, (float, int)):
        vmins: list[float] = [vmin for _ in range(len(profiles))]
    else:
        vmins: list[float] = _vmins

    if cbar_mode == "single":
        vmaxs: list[float] = [max(vmaxs) for _ in range(len(vmaxs))]
        vmins: list[float] = [min(vmins) for _ in range(len(vmins))]

    # import phix raytransfer object
    if rtc is None:
        world = World()
        rtc = import_phix_rtc(world)

    # set image extent
    extent = (
        rtc.material.rmin,
        rtc.material.rmin + rtc.material.dr * rtc.material.grid_shape[0],
        rtc.transform[2, 3],
        -1 * rtc.transform[2, 3],
    )

    # set mask array
    mask = np.logical_not(rtc.mask.squeeze())

    # show 2D profile
    for i, profile in enumerate(profiles):
        # set norm
        norm = set_norm(plot_mode, vmins[i], vmaxs[i], linear_width=linear_width)

        # maske profile out of limiter
        profile = np.ma.masked_array(profile, mask)

        # imshow
        grids[i].imshow(np.transpose(profile), origin="lower", extent=extent, cmap=cmap, norm=norm)

        # plot edge of Outer/Inner Limitter
        grids[i].plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], "k")
        grids[i].plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1], "k")

        # set each axis properties
        set_axis_properties(grids[i])

    # set axis labels
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("$Z$ [m]")
    for i in range(ncol):
        grids[i + (nrow - 1) * ncol].set_xlabel("$R$ [m]")

    # create colorbar objects and store them into a list
    cbars = []
    if cbar_mode == "each":
        for i, grid in enumerate(grids):
            extend = _set_cbar_extend(vmins[i], vmaxs[i], _vmins[i], _vmaxs[i])
            cbar = plt.colorbar(grid.images[0], grids.cbar_axes[i], extend=extend)
            set_cbar_format(cbar.ax, plot_mode, linear_width=linear_width)
            cbars.append(cbar)

    else:  # cbar_mode == "single"
        vmax, vmin = max(vmaxs), min(vmins)
        _vmax, _vmin = max(_vmaxs), min(_vmins)
        extend = _set_cbar_extend(vmin, vmax, _vmin, _vmax)
        norm = set_norm(plot_mode, vmins[0], vmaxs[0], linear_width=linear_width)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)
        set_cbar_format(cbar.ax, plot_mode, linear_width=linear_width)
        cbars.append(cbar)

    # set colorbar label at the last cax
    cbars[-1].set_label(clabel)

    return (fig, grids)


def show_phix_profile(
    axes: Axes,
    profile: NDArray,
    cmap: str = "inferno",
    rtc: RayTransferCylinder | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    plot_contour: bool = True,
    levels: NDArray | None = None,
    plot_mode: str = "scalar",
    linear_width: float = 1.0,
) -> list[NDArray] | None:
    """Show in-phix-limiter 2D profile with :obj:`~matplotlib.axes.Axes.pcolormesh` and plot their
    contours.

    Parameters
    ----------
    axes
        matplotlib Axes object
    profile
         2D-array-like (nr, nz) profile inner PHiX limiter
    cmap
        color map, by default ``"inferno"``
    rtc
        cherab's raytransfer objects, by default the instance loaded by `.import_phix_rtc`.
    vmax
        to set the upper color limitation, by default maximum value of the profile.
    vmin
        to set the lower color limitation, by default minimal value of the profile.
    plot_contour
        whether or not to show contours as well as pcolormesh, by default True
    levels : 1D array-like, optional
        contour's level array, by default 1D array having 10 numbers in range of 0 to the maximum value
    plot_mode
        change the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``, ``"symlog"``, ``"asinh"``},
        by default ``"scalar"``.
        Each mode corresponds to the :obj:`~matplotlib.colors.Normalize` object as follows.
    linear_width
        linear width of asinh/symlog norm, by default 1.0

    Returns
    -------
    list[NDArray] | None
        list of contour line array :math:`(R, Z)` if `plot_contour` is True.
    """
    # set axes option
    axes.set_aspect("equal")

    # import phix raytransfer object
    if rtc is None:
        world = World()
        rtc = import_phix_rtc(world)

    # RZ grid
    z = np.linspace(-1 * rtc.transform[2, 3], rtc.transform[2, 3], rtc.material.grid_shape[2])
    r = np.linspace(
        rtc.material.rmin,
        rtc.material.rmin + rtc.material.dr * rtc.material.grid_shape[0],
        rtc.material.grid_shape[0],
    )
    rr, zz = np.meshgrid(r, z)

    # set vmax, vmin
    vmax = np.asarray_chkfinite(profile).max() if vmax is None else vmax
    vmin = np.asarray_chkfinite(profile).min() if vmin is None else vmin

    # set norm
    norm = set_norm(plot_mode, vmin, vmax, linear_width=linear_width)

    # set masked profile
    mask = np.logical_not(rtc.mask.squeeze())
    profile = np.ma.masked_array(profile, mask)

    # show pcolormesh
    axes.pcolormesh(rr, zz, np.flipud(profile.T), cmap=cmap, norm=norm, shading="auto")

    # plot contour
    if plot_contour:
        # set contour levels
        if levels is None:
            levels = np.linspace(0.0, vmax, 8)

        lines = []
        for level in levels[1:]:
            lines += calc_contours(profile, level, rtc=rtc, r=r, z=z)

        for line in lines:
            axes.plot(line[:, 0], line[:, 1], color="w", linewidth=1)
    else:
        lines = None

    # plot edge of Outer/Inner Limitter
    axes.plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], "k")
    axes.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1], "k")

    # set each axis properties
    set_axis_properties(axes)

    return lines


def set_axis_properties(axes: Axes) -> Axes:
    """Set x-, y-axis property.

    This function set axis properties such as tick direction, tick size, and so on.
    The minor tick interval is set to 0.05.

    Parameters
    ----------
    axes
        matplotlib Axes object

    Returns
    -------
    :obj:`~matplotlib.axes.Axes`
        axes object with new properties
    """
    axes.xaxis.set_minor_locator(MultipleLocator(0.05))
    axes.yaxis.set_minor_locator(MultipleLocator(0.05))
    axes.tick_params(direction="in", labelsize=10, which="both", top=True, right=True)

    return axes


def set_cbar_format(cax: CbarAxesBase | Axes, formatter: str, linear_width: float = 1.0, **kwargs):
    """Set colorbar's locator and formatter.

    Parameters
    ----------
    cax
        colorbar axes object
    formatter
        formatter for colorbar yaxis major locator.
        Must select one in {``"scalar"``, ``"log"``, ``"symlog"``, ``"asinh"``, ``percent``, ``eng``}
    linear_width
        linear width of asinh/symlog norm, by default 1.0
    **kwargs
        keyword arguments for formatter

    Returns
    -------
    Colorbar
        colorbar object with new properties
    """
    # define colobar formatter and locator
    if formatter == "log":
        fmt = LogFormatterSciNotation(**kwargs)
        major_locator = LogLocator(base=10, numticks=None)
        minor_locator = LogLocator(base=10, subs=tuple(np.arange(0.1, 1.0, 0.1)), numticks=12)

    elif formatter == "symlog":
        fmt = LogFormatterSciNotation(linthresh=linear_width, **kwargs)
        major_locator = SymmetricalLogLocator(linthresh=linear_width, base=10)
        minor_locator = SymmetricalLogLocator(
            linthresh=linear_width, base=10, subs=tuple(np.arange(0.1, 1.0, 0.1))
        )

    elif formatter == "asinh":
        raise NotImplementedError("asinh mode is not supported yet (due to old matplotlib).")

    elif formatter == "percent":
        fmt = PercentFormatter(**kwargs)
        major_locator = AutoLocator()
        minor_locator = AutoMinorLocator()

    elif formatter == "eng":
        fmt = EngFormatter(**kwargs)
        major_locator = AutoLocator()
        minor_locator = AutoMinorLocator()

    else:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        major_locator = AutoLocator()
        minor_locator = AutoMinorLocator()

    # set colorbar's locator and formatter
    cax.yaxis.set_offset_position("left")
    cax.yaxis.set_major_formatter(fmt)
    cax.yaxis.set_major_locator(major_locator)
    cax.yaxis.set_minor_locator(minor_locator)


def set_norm(mode: str, vmin: float, vmax: float, linear_width: float = 1.0) -> Normalize:
    """Set variouse :obj:`~matplotlib.colors.Normalize` object.

    Parameters
    ----------
    mode
        the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``, ``"symlog"``, ``"asinh"``}
    vmin
        minimum value of the profile.
    vmax
        maximum value of the profile.
    linear_width
        linear width of asinh/symlog norm, by default 1.0

    Returns
    -------
    Normalize
        norm object
    """
    # set norm
    absolute = max(abs(vmax), abs(vmin))
    if mode == "log":
        if vmin <= 0:
            raise ValueError("vmin must be positive value.")
        norm = LogNorm(vmin=vmin, vmax=vmax)

    elif mode == "symlog":
        norm = SymLogNorm(linthresh=linear_width, vmin=-1 * absolute, vmax=absolute)

    elif mode == "centered":
        norm = Normalize(vmin=-1 * absolute, vmax=absolute)

    elif mode == "asinh":
        raise NotImplementedError("asinh norm is not supported yet (due to old matplotlib).")
        # norm = AsinhNorm(linear_width=linear_width, vmin=-1 * absolute, vmax=absolute)

    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def _set_cbar_extend(user_vmin: float, user_vmax: float, data_vmin: float, data_vmax: float) -> str:
    """Set colorbar's extend.

    Parameters
    ----------
    user_vmin
        user defined minimum value.
    user_vmax
        user defined maximum value.
    data_vmin
        minimum value of the profile.
    data_vmax
        maximum value of the profile.

    Returns
    -------
    str
        colorbar's extend.
    """
    if data_vmin < user_vmin:
        if user_vmax < data_vmax:
            extend = "both"
        else:
            extend = "min"
    else:
        if user_vmax < data_vmax:
            extend = "max"
        else:
            extend = "neither"

    return extend


if __name__ == "__main__":
    DIR = (
        Path(__file__).parent.parent.parent.parent
        / "docs"
        / "notebooks"
        / "data"
        / "reconstructed_data"
        / "synthetic"
    )
    profile = np.load(DIR / "reconstructed.npy")

    fig, ax = plt.subplots()
    con = show_phix_profile(ax, profile, cmap="inferno")
    # con = show_phix_profiles(profile, cmap="inferno")
    plt.show()
