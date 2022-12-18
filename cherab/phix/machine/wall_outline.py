"""Resource module for the limiter wall outline."""
import numpy as np
from matplotlib import pyplot as plt

__all__ = ["INNER_LIMITER", "OUTER_LIMITER", "plot_phix_wall_outline"]


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


def plot_phix_wall_outline(style: str = "k"):
    """plot PHiX Limiter and vessel wall polygons.

    Parameters
    ----------
    style
        Line color, by default "k" means black
    """

    plt.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1], style)
    plt.plot(OUTER_LIMITER[:, 0], OUTER_LIMITER[:, 1], style)
    plt.plot(VESSEL_WALL[:, 0], VESSEL_WALL[:, 1], style)
    plt.xlabel("$R$[m]")
    plt.ylabel("$Z$[m]")
    plt.axis("equal")
