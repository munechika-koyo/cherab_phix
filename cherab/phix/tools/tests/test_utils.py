import numpy as np
import pytest
from raysect.optical import World

from cherab.phix.tools.raytransfer import load_rtc
from cherab.phix.tools.utils import calc_contours, profile_1D_to_2D, profile_2D_to_1D


@pytest.fixture
def rtc():
    world = World()
    rtc = load_rtc(world)
    return rtc


@pytest.fixture
def data_1d(rtc):
    return np.arange(rtc.bins)


@pytest.fixture
def data_2d(rtc):
    arr = rtc.voxel_map.squeeze()
    arr[arr < 0] = 0
    return arr


def test_profile_1D_to_2D(rtc, data_1d, data_2d):
    profile2d = profile_1D_to_2D(data_1d, rtc)
    assert profile2d.shape == rtc.material.grid_shape[::2]
    np.testing.assert_allclose(profile2d, data_2d)


def test_profile_2D_to_1D(rtc, data_1d, data_2d):
    profile1d = profile_2D_to_1D(data_2d, rtc)
    assert profile1d.shape == (rtc.bins,)
    np.testing.assert_allclose(profile1d, data_1d)


def test_calc_contours(rtc, data_2d):
    contours = calc_contours(data_2d, 0.5, rtc=rtc)
    assert len(contours) == 3
