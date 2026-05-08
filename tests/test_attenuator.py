from ophyd_async.core import init_devices
import pytest_asyncio
from ophyd_async.testing import *

from cditools.attenuator import AttenuatorBank


pytest_plugins = ('pytest_asyncio',)

@pytest_asyncio.fixture
async def mock_attenuator_bank():
    async with init_devices(mock=True):
        mock_attenuator_bank = AttenuatorBank()
    yield mock_attenuator_bank

def test_find_closest_attenuation(mock_attenuator_bank: AttenuatorBank):
    nearest = mock_attenuator_bank.find_closest_attenuation(0.7)
    assert nearest.attenuation == 0.644

    nearest2 = mock_attenuator_bank.find_closest_attenuation(0.2)
    assert nearest2.attenuation == 0.196

    nearest3 = mock_attenuator_bank.find_closest_attenuation(0.02)
    assert nearest3.attenuation == 0.08

    nearest4 = mock_attenuator_bank.find_closest_attenuation(0.98)
    assert nearest4.attenuation == 1