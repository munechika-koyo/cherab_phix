import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from raysect.optical import World
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.tools.raytransfer import import_phix_rtm
from cherab.phix.machine.wall_outline import OUTER_LIMITER, INNER_LIMITER


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


def show_phix_profile(profiles, fig=None, clabel="", cmap="inferno", rtm=None, vmin=0.0):
    """show in-phix-limiter 2D profile such as emission profile.

    Parameters
    ----------
    profiles : list or 2D-array
         2D-array-like (nr, nz) profile inner PHiX limiter
         if you want to show several profiles, you must put one list containing 2D array-like data.
    fig : object, optional
        matplotlib figure object
    clabel : str, optional
        colobar label
    cmap : str, optional
        color map, by default "inferno"
    rtm : cherab.raytransfer object, optional
        cherab's raytransfer objects, by default phix's one using TSCEquilibrium()
    vmin : float, optional
        imshow's vmin argument

    Returns
    -------
    (fig, grid) : tuple
        one tuple containing matplotlib objects (figure, ImageGrid)
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
    grid = ImageGrid(fig, 111, (1, len(profiles)), cbar_mode="single", cbar_pad=0)

    # define some valuables in advance
    vmax = max([np.asarray(profile).max() for profile in profiles])
    extent = (
        rtm.material.rmin,
        rtm.material.rmin + rtm.material.dr * rtm.material.grid_shape[0],
        rtm.transform[2, 3],
        -1 * rtm.transform[2, 3],
    )
    xpos = np.array([0.362, 0.42, 0.42])
    ypos = np.array([-0.165, -0.165, -0.06])
    ypos_up = np.array([0.166, 0.166, 0.06])

    # show 2D profile
    imgs = []
    for i, profile in enumerate(profiles):
        imgs.append(
            grid[i].imshow(
                np.transpose(profile),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
            )
        )

        # fill the outer in-limiter to white
        grid[i].fill(xpos, ypos, color="w", alpha=1.0)
        grid[i].fill(xpos, ypos_up, color="w", alpha=1.0)

        # plot edge of OUTER LIMITER
        grid[i].plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], "k")
        grid[i].plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1], "w")

        # axis label
        grid[i].set_xlabel("$R$[m]")

    grid[0].set_ylabel("$Z$[m]")

    # colobar
    cbar = grid.cbar_axes[0].colorbar(imgs[-1])
    cbar.ax.xaxis.set_visible(False)
    cbar.set_label_text(clabel)

    return (fig, grid)
