from __future__ import annotations

import pytest
import pytest_asyncio
from ophyd_async.core import get_mock_put, init_devices, set_mock_value

from cditools.attenuator import AVAILABLE_ATTENUATIONS, AttenuatorBank, AttenuatorEnum

pytest_plugins = ("pytest_asyncio",)


@pytest_asyncio.fixture
async def mock_attenuator_bank():
    async with init_devices(mock=True):
        mock_attenuator_bank = AttenuatorBank()
    yield mock_attenuator_bank


@pytest.mark.asyncio
async def test_set_attenuators(mock_attenuator_bank: AttenuatorBank):
    atten_mock0 = get_mock_put(mock_attenuator_bank.attenuators[0].cmd)
    atten_mock1 = get_mock_put(mock_attenuator_bank.attenuators[1].cmd)
    atten_mock2 = get_mock_put(mock_attenuator_bank.attenuators[2].cmd)
    atten_mock3 = get_mock_put(mock_attenuator_bank.attenuators[3].cmd)

    # AttenuatorCombination(attenuation=0.095, attenuators=[1, 2, 3]),
    combo0 = AVAILABLE_ATTENUATIONS[1]  # attenuators 1,2,3
    await mock_attenuator_bank.set_attenuation(combo0.attenuation)
    atten_mock0.assert_called_with(AttenuatorEnum.LOW)
    atten_mock1.assert_called_with(AttenuatorEnum.HIGH)
    atten_mock2.assert_called_with(AttenuatorEnum.HIGH)
    atten_mock3.assert_called_with(AttenuatorEnum.HIGH)

    # AttenuatorCombination(attenuation=0.768, attenuators=[1]),
    combo1 = AVAILABLE_ATTENUATIONS[-3]
    await mock_attenuator_bank.set_attenuation(combo1.attenuation)
    atten_mock0.assert_called_with(AttenuatorEnum.LOW)
    atten_mock1.assert_called_with(AttenuatorEnum.HIGH)
    atten_mock2.assert_called_with(AttenuatorEnum.LOW)
    atten_mock3.assert_called_with(AttenuatorEnum.LOW)


@pytest.mark.asyncio
async def test_get_bank_status(mock_attenuator_bank: AttenuatorBank):
    set_mock_value(mock_attenuator_bank.attenuators[0].status, AttenuatorEnum.LOW)
    set_mock_value(mock_attenuator_bank.attenuators[1].status, AttenuatorEnum.LOW)
    set_mock_value(mock_attenuator_bank.attenuators[2].status, AttenuatorEnum.HIGH)
    set_mock_value(mock_attenuator_bank.attenuators[3].status, AttenuatorEnum.LOW)

    assert await mock_attenuator_bank.get_status() == [
        AttenuatorEnum.LOW,
        AttenuatorEnum.LOW,
        AttenuatorEnum.HIGH,
        AttenuatorEnum.LOW,
    ]


def test_find_closest_attenuation(mock_attenuator_bank: AttenuatorBank):
    nearest = mock_attenuator_bank.find_closest_attenuation(0.7)
    assert nearest.attenuation == 0.644

    nearest2 = mock_attenuator_bank.find_closest_attenuation(0.2)
    assert nearest2.attenuation == 0.196

    nearest3 = mock_attenuator_bank.find_closest_attenuation(0.02)
    assert nearest3.attenuation == 0.08

    nearest4 = mock_attenuator_bank.find_closest_attenuation(0.98)
    assert nearest4.attenuation == 1


def test_up_to_date_available_attenuations(mock_attenuator_bank: AttenuatorBank):
    assert (
        mock_attenuator_bank._calculate_available_attentuations()  # type: ignore[reportPrivateUsage]
        == AVAILABLE_ATTENUATIONS
    )
