from contextlib import nullcontext as does_not_raise

import pytest
from raysect.optical import World

from cherab.core.plasma import Plasma
from cherab.phix.plasma.plasma import load_plasma
from cherab.tools.equilibrium import EFITEquilibrium


@pytest.mark.parametrize(
    ["eq_model", "expectation"],
    [
        pytest.param("phix10", does_not_raise(), id="phix10"),
        pytest.param("phix11", pytest.raises(FileNotFoundError), id="phix11"),
        pytest.param("phix12", does_not_raise(), id="phix12"),
        pytest.param("phix13", does_not_raise(), id="phix13"),
        pytest.param("phix14", does_not_raise(), id="phix14"),
    ],
)
def test_load_plasma(eq_model, expectation):
    with expectation:
        world = World()
        plasma, eq = load_plasma(world, eq_model=eq_model)
        assert isinstance(plasma, Plasma)
        assert isinstance(eq, EFITEquilibrium)
