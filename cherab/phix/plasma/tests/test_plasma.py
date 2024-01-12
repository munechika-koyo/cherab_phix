from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from raysect.optical import World

from cherab.core.plasma import Plasma
from cherab.phix.plasma.plasma import emission_hydrogen_balmer, load_plasma
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


@pytest.mark.parametrize(
    ["r", "z", "kwargs", "expectation"],
    [
        pytest.param([0.34], [0.0], {}, does_not_raise(), id="default"),
        pytest.param([0.34], [0.0], {"balmer": "alpha"}, does_not_raise(), id="balmer-alpha"),
        pytest.param([0.34], [0.0], {"balmer": "beta"}, does_not_raise(), id="balmer-beta"),
        pytest.param([0.34], [0.0], {"balmer": "gamma"}, does_not_raise(), id="balmer-gamma"),
        pytest.param([0.34], [0.0], {"balmer": "delta"}, does_not_raise(), id="balmer-delta"),
        pytest.param([0.34, 0.35], [0.0, 0.05], {}, does_not_raise(), id="multiple-points"),
        pytest.param(
            np.linspace(0.25, 0.42, 10),
            np.linspace(-0.15, 0.115, 10),
            {},
            does_not_raise(),
            id="multiple-points (numpy array)",
        ),
        pytest.param("invalid", [0.0], {}, pytest.raises(TypeError), id="invalid type grid"),
        pytest.param(
            [0.34], [[0.0], [0.1]], {}, pytest.raises(ValueError), id="invalid grid dimension"
        ),
        pytest.param([0.34], [0.0], {"spectrum_step": 0.01}, does_not_raise(), id="spectrum-step"),
        pytest.param(
            [0.34],
            [0.0],
            {"spectrum_step": 0.0},
            pytest.raises(ValueError),
            id="invalid spectrum-step",
        ),
        pytest.param([0.34], [0.0], {"eq_model": "phix12"}, does_not_raise(), id="eq-model"),
    ],
)
def test_emission_hydrogen_balmer(r, z, kwargs, expectation):
    with expectation:
        emiss = emission_hydrogen_balmer(r, z, **kwargs)
        assert emiss.shape == (len(r), len(z))
