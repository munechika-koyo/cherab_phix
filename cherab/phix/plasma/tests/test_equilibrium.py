from contextlib import nullcontext as does_not_raise

import pytest

from cherab.phix.plasma.equilibrium import load_equilibrium
from cherab.tools.equilibrium import EFITEquilibrium


@pytest.mark.parametrize(
    ["model_variant", "expectation"],
    [
        pytest.param("phix10", does_not_raise(), id="phix10"),
        pytest.param("phix11", pytest.raises(FileNotFoundError), id="phix11"),
        pytest.param("phix12", does_not_raise(), id="phix12"),
        pytest.param("phix13", does_not_raise(), id="phix13"),
        pytest.param("phix14", does_not_raise(), id="phix14"),
    ],
)
def test_load_equilibrium(model_variant, expectation):
    with expectation:
        equilibrium = load_equilibrium(model_variant)
        assert isinstance(equilibrium, EFITEquilibrium)
