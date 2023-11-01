"""Resource module for the limiter wall outline."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["INNER_LIMITER", "OUTER_LIMITER", "plot_wall_outline"]


INNER_LIMITER = np.array(
    [
        [0.2400, 0.1650],
        [0.2400, -0.1650],
        [0.3620, -0.1650],
        [0.4200, -0.0600],
        [0.4200, 0.0600],
        [0.3620, 0.1650],
        [0.2400, 0.1650],
    ]
)
OUTER_LIMITER = np.array(
    [
        [0.22050, -0.18],
        [0.22050, 0.18],
        [0.41746, 0.18],
        [0.43650, 0.146675],
        [0.43650, -0.146675],
        [0.41746, -0.18],
        [0.22050, -0.18],
    ]
)

VESSEL_WALL = np.array(
    [[0.220, -0.185], [0.220, 0.185], [0.440, 0.185], [0.440, -0.185], [0.220, -0.185]]
)


def plot_wall_outline(ax: Axes | None = None, **kwargs) -> tuple[Figure, Axes] | Axes:
    """Plot PHiX Limiter and vessel wall polygons.

    Parameters
    ----------
    ax
        The axes to plot on. If None, a new figure and axes is created and returned, by default None.
    kwargs
        Keyword arguments passed to `matplotlib.pyplot.plot` for each polygon.

    Returns
    -------
    tuple[:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`] | :obj:`matplotlib.axes.Axes`
        The figure and axes used to plot the polygons.
        If `ax` is not None, only the axes is returned.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from cherab.phix.machine import plot_wall_outline
        >>> fig, ax = plot_wall_outline(color="k")
        >>> fig.show()

    .. image:: ../_static/images/plots/plot_wall_outline.png
    """
    if ax is None:
        fig, ax = plt.subplots()
        _axes_none = True
    else:
        _axes_none = False

    ax.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1], **kwargs)
    ax.plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], **kwargs)
    ax.plot(VESSEL_WALL[:, 0], VESSEL_WALL[:, 1], **kwargs)

    if _axes_none:
        ax.set_xlabel("$R$[m]")
        ax.set_ylabel("$Z$[m]")
        ax.set_aspect(1)
        return fig, ax

    else:
        return ax
