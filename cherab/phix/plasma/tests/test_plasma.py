from contextlib import nullcontext as does_not_raise

import pytest
from raysect.optical import World

from cherab.core.plasma import Plasma
from cherab.phix.plasma.plasma import load_plasma
from cherab.tools.equilibrium import EFITEquilibrium


@pytest.mark.parametrize(
    ["equilibrium", "expectation"],
    [
        pytest.param("phix10", does_not_raise(), id="phix10"),
        pytest.param("phix11", pytest.raises(FileNotFoundError), id="phix11"),
        pytest.param("phix12", does_not_raise(), id="phix12"),
        pytest.param("phix13", does_not_raise(), id="phix13"),
        pytest.param("phix14", does_not_raise(), id="phix14"),
    ],
)
def test_load_plasma(equilibrium, expectation):
    with expectation:
        world = World()
        plasma, eq = load_plasma(world, equilibrium=equilibrium)
        assert isinstance(plasma, Plasma)
        assert isinstance(eq, EFITEquilibrium)
