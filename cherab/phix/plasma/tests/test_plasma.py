import pytest
from raysect.optical import World

from cherab.core.plasma import Plasma
from cherab.phix.plasma.plasma import import_plasma
from cherab.tools.equilibrium import EFITEquilibrium


@pytest.mark.parametrize(
    ["equilibrium"],
    [
        pytest.param("phix10", id="phix10"),
        pytest.param("phix12", id="phix12"),
        pytest.param("phix13", id="phix13"),
        pytest.param("phix14", id="phix14"),
    ],
)
def test_import_plasma(equilibrium):
    world = World()
    plasma, eq = import_plasma(world, equilibrium=equilibrium)
    assert isinstance(plasma, Plasma)
    assert isinstance(eq, EFITEquilibrium)
