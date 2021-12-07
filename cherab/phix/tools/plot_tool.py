import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter, ScalarFormatter, LinearLocator
from mpl_toolkits.axes_grid1 import ImageGrid
from raysect.optical import World
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.tools.raytransfer import import_phix_rtm
from cherab.phix.machine.wall_outline import OUTER_LIMITER


def plot_ray(ray, world=None):
    """plotting the spectrum of one ray-tracing

    Parameters
    ----------
    ray : :obj:`~raysect.core.ray.Ray`
        raysect Ray object
    world : :obj:`~raysect.core.scenegraph.node.Node`, optional
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
    profiles,
    fig=None,
    clabel="",
    cmap="inferno",
    rtm=None,
    vmax=None,
    vmin=None,
    axes_pad=0.02,
    cbar_mode="single",
    scientific_notation=False,
    plot_mode="scalar",
):
    """show in-phix-limiter 2D profiles such as emission profile.
    This function can show several 2D profiles with matplotlib  imshow style

    Parameters
    ----------
    profiles : list or 2D-array
         2D-array-like (nr, nz) profile inner PHiX limiter
         if you want to show several profiles, you must put one list containing 2D array-like data.
    fig : matplotlib.figure.Figure, optional
        matplotlib figure object
    clabel : str, optional
        colobar label
    cmap : str, optional
        color map, by default "inferno"
    rtm : :obj:`~cherab.tools.raytransfer.raytransfer.RayTransferCylinder`, optional
        cherab's raytransfer objects, by default returns using default`.TSCEquilibrium` and `.import_phix_rtm`
    vmax : float, optional
        to set the upper color limitation, by default maximum value of all profiles, if cbar_mode=="single"
    vmin : float, optional
        to set the lower color limitation, by default minimal value of all profiles, if cbar_mode=="single"
    axes_pad : float, optional
        ImageGrid's para to set the interval between axes, by default 0.02
    cbar_mode : str, optional
        ImgeGrid's para to set colorbars in "single" axes or "each" axes, by default "single"
    scientific_notation : bool, optional
        whether or not to set colorbar fomat with scientific notation, by default Flase
    plot_mode : str, optional
        plot 2D mesh as "scalor" or "log", by default "scalar"

    Returns
    -------
    (fig, grids) : tuple
        one tuple containing matplotlib objects (:obj:`~matplotlib.figure.Figure`, :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`)
    """
    # transform the type of argument if it is not list type.
    if not isinstance(profiles, list):
        profiles = [profiles]

    # import phix raytransfer object
    if rtm is None:
        world = World()
        eq = TSCEquilibrium()
        rtm = import_phix_rtm(world, equilibrium=eq)

    # set ImageGrid
    fig = fig or plt.figure()
    grids = ImageGrid(fig, 111, (1, len(profiles)), axes_pad=axes_pad, cbar_mode=cbar_mode, cbar_pad=0.0)

    # define some valuables in advance
    if vmax is None:
        vmax = [np.asarray(profile).max() for profile in profiles]
    if vmin is None:
        vmin = [np.asarray(profile).min() for profile in profiles]
    if cbar_mode == "single":
        vmax = [np.max(vmax) for i in range(len(profiles))]
        vmin = [np.min(vmin) for i in range(len(profiles))]

    # set image extent
    extent = (
        rtm.material.rmin,
        rtm.material.rmin + rtm.material.dr * rtm.material.grid_shape[0],
        rtm.transform[2, 3],
        -1 * rtm.transform[2, 3],
    )

    # set mask array
    mask = np.logical_not(rtm.mask.squeeze())

    # show 2D profile
    for i, profile in enumerate(profiles):
        # color limit
        if plot_mode == "log":
            if min(vmin) <= 0:
                raise ValueError("profile must not have 0 or less.")

            norm = LogNorm(vmin=vmin[i], vmax=vmax[i])
        else:
            norm = Normalize(vmin=vmin[i], vmax=vmax[i])

        # maske profile out of limiter
        profile = np.ma.masked_array(profile, mask)
        # imshow
        grids[i].imshow(np.transpose(profile), origin="lower", extent=extent, cmap=cmap, norm=norm)

        # plot edge of OUTER LIMITER
        grids[i].plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], "k")

        # axis label
        grids[i].set_xlabel("$R$[m]")

    grids[0].set_ylabel("$Z$[m]")

    # colobar
    # TODO: need to find out why the first colorbar does not show with scientific notation
    # when show more than 2 gird images.
    if plot_mode == "log":
        fmt = LogFormatterSciNotation()
        major_locator = LogLocator(base=10, numticks=None)
        # minor_locator = LogLocator(base=10)
        minor_locator = LogLocator(base=10, subs=tuple(np.arange(0.1, 1.0, 0.1)), numticks=12)
    else:
        fmt = ScalarFormatter(useMathText=True)
        major_locator = LinearLocator()
        minor_locator = LinearLocator()

    if scientific_notation is True:
        fmt.set_powerlimits((0, 0))

    if cbar_mode == "each":
        cbar = None  # initialization
        for i, grid in enumerate(grids):
            cbar = grids.cbar_axes[i].colorbar(grid.images[0])
            cbar.cbar_axis.set_major_locator(major_locator)
            cbar.cbar_axis.set_minor_locator(minor_locator)
            cbar.cbar_axis.set_major_formatter(fmt)
            cbar.cbar_axis.set_minor_formatter(NullFormatter())
            cbar.ax.xaxis.set_visible(False)
            cbar.ax.yaxis.set_offset_position("left")
        cbar.set_label_text(clabel)
    else:
        cbar = grids.cbar_axes[0].colorbar(grids[-1].images[0])
        cbar.cbar_axis.set_major_locator(major_locator)
        cbar.cbar_axis.set_minor_locator(minor_locator)
        cbar.cbar_axis.set_major_formatter(fmt)
        cbar.cbar_axis.set_minor_formatter(NullFormatter())
        cbar.ax.xaxis.set_visible(False)
        cbar.ax.yaxis.set_offset_position("left")
        cbar.ax.xaxis.set_visible(False)
        cbar.ax.yaxis.set_offset_position("left")
        cbar.set_label_text(clabel)

    return (fig, grids)


def show_phix_profile(
    axes,
    profile,
    cmap="inferno",
    rtm=None,
    vmax=None,
    vmin=None,
    toggle_contour=True,
    levels=None,
    scientific_notation=False,
):
    """show phix one in-limiter profile using pcolormesh and contour if you want.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        matplotlib Axes object
    profiles : list or 2D-array
         2D-array-like (nr, nz) profile inner PHiX limiter
    cmap : str, optional
        color map, by default "inferno"
    rtm : :obj:`~cherab.tools.raytransfer.raytransfer.RayTransferCylinder`, optional
        cherab's raytransfer objects, by default returns using default`.TSCEquilibrium` and `.import_phix_rtm`
    vmax : float, optional
        to set the upper color limitation, by default maximum value of all profiles, if cbar_mode=="single"
    vmin : float, optional
        to set the lower color limitation, by default minimal value of all profiles, if cbar_mode=="single"
    toggle_contour : bool, optional
        toggle whether or not to show contour plot as well as pcolormesh, by default True
    levels : 1D array-like, optional
        contour's level array, by default number of levels is 10 in range of 0 to max value
    scientific_notation : bool, optional
        whether or not to set colorbar fomat with scientific notation, by default Flase

    Returns
    ------
    :obj:`~matplotlib.contour.QuadContourSet`
        matplotlib contour object
    """
    # set axes option
    axes.set_aspect("equal")

    # import phix raytransfer object
    if rtm is None:
        world = World()
        eq = TSCEquilibrium(folder="phix10")
        rtm = import_phix_rtm(world, equilibrium=eq)

    # RZ grid
    z = np.linspace(-1 * rtm.transform[2, 3], rtm.transform[2, 3], rtm.material.grid_shape[2])
    r = np.linspace(
        rtm.material.rmin, rtm.material.rmin + rtm.material.dr * rtm.material.grid_shape[0], rtm.material.grid_shape[0]
    )
    R, Z = np.meshgrid(r, z)

    # set vmax, vmin
    if vmax is None:
        vmax = np.asarray_chkfinite(profile).max()
    if vmin is None:
        vmin = np.asarray_chkfinite(profile).min()

    # set contour levels
    if levels is None:
        levels = np.linspace(0, vmax, 8)

    # set masked profile
    mask = np.logical_not(rtm.mask.squeeze())
    profile = np.ma.masked_array(profile, mask)
    # show pcolormesh
    axes.pcolormesh(R, Z, np.flipud(profile.T), cmap=cmap, vmax=vmax, vmin=vmin, shading="auto")

    # plot contour
    if toggle_contour is True:
        contour = axes.contour(R, Z, np.flipud(profile.T), levels, colors="w", linewidths=1)
    else:
        contour = None

    # plot edge of OUTER LIMITER
    axes.plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], "k")
    # axis label
    # axes.set_xlabel("$R$[m]")
    # axes.set_ylabel("$Z$[m]")

    return contour


if __name__ == "__main__":
    DIR = os.path.dirname(__file__).split("/cherab/")[0]
    profile = np.load(
        os.path.join(DIR, "output", "data", "experiment", "shot_17393", "camera_data", "reconstraction", "1060.npy")
    )

    fig, ax = plt.subplots()
    con = show_phix_profile(ax, profile, cmap="inferno")
    plt.show()
