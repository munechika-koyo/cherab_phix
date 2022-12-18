"""Modules to offer visualization tools."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from cherab.tools.raytransfer.raytransfer import RayTransferCylinder
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import (
    AutoLocator,
    AutoMinorLocator,
    LogFormatterSciNotation,
    LogLocator,
    ScalarFormatter,
)
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.typing import NDArray
from raysect.optical import Ray, World

from cherab.phix.machine.wall_outline import INNER_LIMITER, OUTER_LIMITER
from cherab.phix.tools.raytransfer import import_phix_rtc
from cherab.phix.tools.utils import calc_contours

__all__ = ["plot_ray", "show_phix_profiles", "show_phix_profile"]


def plot_ray(ray: Ray, world: World):
    """plotting the spectrum of one ray-tracing.

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
    clabel: str = "",
    cmap: str = "inferno",
    rtc: RayTransferCylinder | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    axes_pad: float = 0.02,
    cbar_mode: str = "single",
    scientific_notation: bool = True,
    plot_mode: str = "scalar",
) -> tuple[Figure, ImageGrid]:
    """show in-phix-limiter 2D profiles such as emission profile.

    This function can show several 2D profiles with matplotlib imshow style.

    Parameters
    ----------
    profiles
         2D-array-like (nr, nz) profile inner PHiX limiter
         if you want to show several profiles, you must put one list containing 2D array-like data.
    fig
        Figure object, by default `plt.figure()`
    clabel
        colobar label
    cmap
        color map, by default "inferno"
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
        change the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``}, by default ``"scalar"``

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

    grids = ImageGrid(
        fig,
        111,
        (1, len(profiles)),
        axes_pad=axes_pad,
        cbar_mode=cbar_mode,
        cbar_pad=0.0,
    )

    # define vmaxs
    if isinstance(vmax, (float, int)):
        vmaxs: list[float] = [vmax for _ in range(len(profiles))]
    else:
        vmaxs: list[float] = [profile.max() for profile in profiles]

    # define vmins
    if isinstance(vmin, (float, int)):
        vmins: list[float] = [vmin for _ in range(len(profiles))]
    else:
        vmins: list[float] = [profile.min() for profile in profiles]

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
        # color limit
        if plot_mode == "log":
            if min(vmins) <= 0:
                raise ValueError("profile must not have 0 or less.")

            norm = LogNorm(vmin=vmins[i], vmax=vmaxs[i])
        elif plot_mode == "centered":
            norm = CenteredNorm(vcenter=0, halfrange=max(abs(vmaxs[i]), abs(vmins[i])))
        else:
            norm = Normalize(vmin=vmins[i], vmax=vmaxs[i])

        # maske profile out of limiter
        profile = np.ma.masked_array(profile, mask)
        # imshow
        grids[i].imshow(np.transpose(profile), origin="lower", extent=extent, cmap=cmap, norm=norm)

        # plot edge of Outer/Inner Limitter
        grids[i].plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], "k")
        grids[i].plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1], "k")

        # axis label
        grids[i].set_xlabel("$R$[m]")

    grids[0].set_ylabel("$Z$[m]")

    # create colorbar objects and store them into a list
    cbars = []
    if cbar_mode == "each":
        for i, grid in enumerate(grids):
            cbar = plt.colorbar(grid.images[0], grids.cbar_axes[i])
            cbars.append(cbar)
    else:
        cbar = plt.colorbar(grids[-1].images[0], grids.cbar_axes[0])
        cbars.append(cbar)

    # define colobar formatter
    if plot_mode == "log":
        fmt = LogFormatterSciNotation()
    else:
        fmt = ScalarFormatter(useMathText=True)

    if scientific_notation and plot_mode != "log":
        fmt.set_powerlimits((0, 0))

    # set colorbar's locator and formatter
    for cbar in cbars:
        cbar.ax.yaxis.set_offset_position("left")

        if plot_mode == "log":
            cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=None))
            cbar.ax.yaxis.set_minor_locator(
                LogLocator(base=10, subs=tuple(np.arange(0.1, 1.0, 0.1)), numticks=12)
            )
            cbar.ax.yaxis.set_major_formatter(fmt)
        else:
            cbar.ax.yaxis.set_major_locator(AutoLocator())
            cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
            cbar.ax.yaxis.set_major_formatter(fmt)

    # set colorbar label at the last cax
    cbars[-1].set_label(clabel)

    return (fig, grids)


def show_phix_profile(
    axes: Axes,
    profile: list[NDArray] | NDArray,
    cmap: str = "inferno",
    rtc: RayTransferCylinder | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    toggle_contour: bool = True,
    levels: NDArray | None = None,
    plot_mode: str = "scalar",
) -> list[NDArray] | None:
    """show in-phix-limiter 2D profile with
    :obj:`~matplotlib.axes.Axes.pcolormesh` and plot their contours.

    Parameters
    ----------
    axes
        matplotlib Axes object
    profiles
         2D-array-like (nr, nz) profile inner PHiX limiter
    cmap
        color map, by default "inferno"
    rtc
        cherab's raytransfer objects, by default the instance loaded by `.import_phix_rtc`.
    vmax
        to set the upper color limitation, by default maximum value of all profiles,
        if ``cbar_mode=="single"``
    vmin
        to set the lower color limitation, by default minimal value of all profiles,
        if ``cbar_mode=="single"``
    toggle_contour
        whether or not to show contours as well as pcolormesh, by default True
    levels : 1D array-like, optional
        contour's level array, by default 1D array having 10 numbers in range of 0 to the maximum value
    plot_mode
        change the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``}, by default ``"scalar"``

    Returns
    -------
    list[NDArray] | None
        list of contour line array :math:`(R, Z)` if `toggle_contour` is True.
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
    if vmax is None:
        vmax = np.asarray_chkfinite(profile).max()
    if vmin is None:
        vmin = np.asarray_chkfinite(profile).min()

    if plot_mode == "scaler":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif plot_mode == "centered":
        norm = CenteredNorm(vcenter=0.0, halfrange=max(abs(vmax), abs(vmin)))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # set masked profile
    mask = np.logical_not(rtc.mask.squeeze())
    profile = np.ma.masked_array(profile, mask)

    # show pcolormesh
    axes.pcolormesh(rr, zz, np.flipud(profile.T), cmap=cmap, norm=norm, shading="auto")

    # plot contour
    if toggle_contour is True:
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

    return lines


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
