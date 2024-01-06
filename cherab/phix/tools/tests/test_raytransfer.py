import pytest
from raysect.optical import World

from cherab.phix.plasma.equilibrium import import_equilibrium
from cherab.phix.tools.raytransfer import import_phix_rtc
from cherab.tools.raytransfer.raytransfer import RayTransferCylinder

DEFAULT_EQ = import_equilibrium(model_variant="phix12")


@pytest.mark.parametrize(
    ["equilibrium"],
    [
        pytest.param(None, id="no_equilibrium"),
        pytest.param(DEFAULT_EQ, id="equilibrium"),
    ],
)
def test_import_phix_rtc(equilibrium):
    world = World()
    rtc = import_phix_rtc(world, equilibrium)
    assert isinstance(rtc, RayTransferCylinder)
    assert rtc.bins == 13326
    assert rtc.material.grid_shape == (90, 1, 165)
    assert rtc.material.grid_steps == (0.002, 360.0, 0.002)