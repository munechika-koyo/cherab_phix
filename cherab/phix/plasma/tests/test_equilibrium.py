import pytest

from cherab.phix.plasma.equilibrium import import_equilibrium
from cherab.tools.equilibrium import EFITEquilibrium


@pytest.mark.parametrize(
    ["model_variant"],
    [
        pytest.param("phix10", id="phix10"),
        pytest.param("phix12", id="phix12"),
        pytest.param("phix13", id="phix13"),
        pytest.param("phix14", id="phix14"),
    ],
)
def test_import_equilibrium(model_variant):
    equilibrium = import_equilibrium(model_variant)
    assert isinstance(equilibrium, EFITEquilibrium)
