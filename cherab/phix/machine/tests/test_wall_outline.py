import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cherab.phix.machine.wall_outline import plot_wall_outline


@pytest.mark.parametrize(
    ["ax", "kwargs"],
    [
        pytest.param(None, {}),
        pytest.param(plt.axes(), {}),
        pytest.param(None, {"color": "k"}),
        pytest.param(None, {"color": "k", "linestyle": "--"}),
        pytest.param(None, {"color": "k", "linestyle": "--", "linewidth": 2}),
    ],
)
def test_plot_wall_outline(ax, kwargs):
    results = plot_wall_outline(ax, **kwargs)
    if isinstance(results, tuple):
        fig, axes = results
        assert isinstance(fig, Figure)
        assert isinstance(axes, Axes)
        fig.show()
    else:
        assert isinstance(results, Axes)
        fig = results.get_figure()
        fig.show()

    plt.close(fig)
