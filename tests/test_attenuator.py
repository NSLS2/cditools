from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from ophyd_async.core import get_mock_put, init_devices, set_mock_value
from ophyd_async.testing import assert_value

from cditools.attenuator import (
    Attenuator,
    AttenuatorBank,
    AttenuatorCombination,
    AttenuatorStatusEnum,
)
from cditools.motors import Energy

pytest_plugins = ("pytest_asyncio",)
photon_energy = 8.6  # KeV
prefix = "test-prefix"
attenuator_configs = [("Al", 16.0), ("Al", 24.0), ("Al", 66.0), ("Al", 124.0)]

# These are the attenuations at photon_energy = 8.6 KeV
TEST_ATTENUATIONS = [
    AttenuatorCombination(transmission=0.084, attenuators=[1, 2, 3, 4]),
    AttenuatorCombination(transmission=0.1, attenuators=[2, 3, 4]),
    AttenuatorCombination(transmission=0.109, attenuators=[1, 3, 4]),
    AttenuatorCombination(transmission=0.129, attenuators=[3, 4]),
    AttenuatorCombination(transmission=0.171, attenuators=[1, 2, 4]),
    AttenuatorCombination(transmission=0.203, attenuators=[2, 4]),
    AttenuatorCombination(transmission=0.222, attenuators=[1, 4]),
    AttenuatorCombination(transmission=0.263, attenuators=[4]),
    AttenuatorCombination(transmission=0.32, attenuators=[1, 2, 3]),
    AttenuatorCombination(transmission=0.38, attenuators=[2, 3]),
    AttenuatorCombination(transmission=0.414, attenuators=[1, 3]),
    AttenuatorCombination(transmission=0.492, attenuators=[3]),
    AttenuatorCombination(transmission=0.65, attenuators=[1, 2]),
    AttenuatorCombination(transmission=0.772, attenuators=[2]),
    AttenuatorCombination(transmission=0.842, attenuators=[1]),
    AttenuatorCombination(transmission=1.0, attenuators=[]),
]


@pytest_asyncio.fixture
async def mock_attenuator_bank():
    async with init_devices(mock=True):
        mock_energy = MagicMock(spec=Energy)
        mock_energy.energy.readback.get.return_value = photon_energy
        mock_energy.egu = "KeV"
        mock_attenuator_bank = AttenuatorBank(prefix, attenuator_configs, mock_energy)
    yield mock_attenuator_bank


@pytest_asyncio.fixture
async def mock_attenuator(mock_attenuator_bank: AttenuatorBank):
    async with init_devices(mock=True):
        mock_attenuator = Attenuator(mock_attenuator_bank.prefix, 1, "Al", 16)
    yield mock_attenuator


class TestAttenuator:
    @pytest.mark.asyncio
    async def test_open(self, mock_attenuator: Attenuator):
        set_mock_value(mock_attenuator.position, AttenuatorStatusEnum.HIGH)
        await mock_attenuator.open()
        await assert_value(mock_attenuator.position, AttenuatorStatusEnum.LOW)

    @pytest.mark.asyncio
    async def test_close(self, mock_attenuator: Attenuator):
        set_mock_value(mock_attenuator.position, AttenuatorStatusEnum.LOW)
        await mock_attenuator.close()
        await assert_value(mock_attenuator.position, AttenuatorStatusEnum.HIGH)

    def test_transmission_kev(self, mock_attenuator: Attenuator):
        assert mock_attenuator.transmission(photon_energy) == pytest.approx(
            0.84, abs=0.01
        )

    def test_transmission_ev(self, mock_attenuator: Attenuator):
        photon_energy = 8600  # eV
        assert mock_attenuator.transmission(photon_energy, egu="eV") == pytest.approx(
            0.84, abs=0.01
        )

    def test_attenuation_kev(self, mock_attenuator: Attenuator):
        assert mock_attenuator.attenuation(photon_energy) == pytest.approx(
            0.16, abs=0.01
        )

    def test_attenuation_ev(self, mock_attenuator: Attenuator):
        photon_energy = 8600  # eV
        assert mock_attenuator.attenuation(photon_energy, egu="eV") == pytest.approx(
            0.16, abs=0.01
        )


class TestAttenuatorBank:
    @pytest.mark.asyncio
    async def test_attenuation_bank_creation(
        self, mock_attenuator_bank: AttenuatorBank
    ):
        assert mock_attenuator_bank.energy.energy.readback.get() == 8.6
        # assert mock_attenuator_bank.photon_energy == 8.6

        second_energy = MagicMock(spec=Energy)
        second_energy.energy.readback.get.return_value = 6
        second_bank = AttenuatorBank(prefix, attenuator_configs, second_energy)
        assert second_bank.energy.energy.readback.get() == 6
        # assert second_bank.photon_energy == 6

    @pytest.mark.asyncio
    async def test_attenuators_indexed_at_1(self, mock_attenuator_bank: AttenuatorBank):
        with pytest.raises(KeyError):
            mock_attenuator_bank.attenuators[0]

        atten1 = mock_attenuator_bank.attenuators[1]
        assert atten1.num == 1
        assert atten1.thickness == 16
        assert atten1.position.source == "mock+ca://test-prefix:DO1-Sts"
        assert atten1.mode.source == "mock+ca://test-prefix:DIO1-Mode"
        assert atten1.in_status.source == "mock+ca://test-prefix:DI1-Sts"
        assert atten1.name == "attenuator_1"

        atten2 = mock_attenuator_bank.attenuators[2]
        assert atten2.num == 2
        assert atten2.thickness == 24

        atten3 = mock_attenuator_bank.attenuators[3]
        assert atten3.num == 3
        assert atten3.thickness == 66

        atten4 = mock_attenuator_bank.attenuators[4]
        assert atten4.num == 4
        assert atten4.thickness == 124

    @pytest.mark.asyncio
    async def test_set_attenuation(self, mock_attenuator_bank: AttenuatorBank):
        atten_mock1 = get_mock_put(mock_attenuator_bank.attenuators[1].position)
        atten_mock2 = get_mock_put(mock_attenuator_bank.attenuators[2].position)
        atten_mock3 = get_mock_put(mock_attenuator_bank.attenuators[3].position)
        atten_mock4 = get_mock_put(mock_attenuator_bank.attenuators[4].position)

        combo0 = TEST_ATTENUATIONS[1]  # attenuators 2, 3, 4
        await mock_attenuator_bank.set(combo0.transmission)
        atten_mock1.assert_called_with(AttenuatorStatusEnum.LOW)
        atten_mock2.assert_called_with(AttenuatorStatusEnum.HIGH)
        atten_mock3.assert_called_with(AttenuatorStatusEnum.HIGH)
        atten_mock4.assert_called_with(AttenuatorStatusEnum.HIGH)

        combo1 = TEST_ATTENUATIONS[-3]  # attenuator 2
        await mock_attenuator_bank.set(combo1.transmission)
        atten_mock1.assert_called_with(AttenuatorStatusEnum.LOW)
        atten_mock2.assert_called_with(AttenuatorStatusEnum.HIGH)
        atten_mock3.assert_called_with(AttenuatorStatusEnum.LOW)
        atten_mock4.assert_called_with(AttenuatorStatusEnum.LOW)

    @pytest.mark.asyncio
    async def test_read(self, mock_attenuator_bank: AttenuatorBank):
        mock_attenuator_bank.set(1)
        status = await mock_attenuator_bank.read()
        assert status == {
            "photon_energy": 8.6,
            "egu": "KeV",
            "total_transmission": 1,
            "attenuator_1": {"active": False, "transmission": 0},
            "attenuator_2": {"active": False, "transmission": 0},
            "attenuator_3": {"active": False, "transmission": 0},
            "attenuator_4": {"active": False, "transmission": 0},
        }

        # Test with different energy and attenuations
        async with init_devices(mock=True):
            second_energy = MagicMock(spec=Energy)
            second_energy.energy.readback.get.return_value = 12
            second_energy.egu = "KeV"
            second_bank = AttenuatorBank(prefix, attenuator_configs, second_energy)
        set_mock_value(second_bank.attenuators[1].position, AttenuatorStatusEnum.LOW)
        set_mock_value(second_bank.attenuators[2].position, AttenuatorStatusEnum.HIGH)
        set_mock_value(second_bank.attenuators[3].position, AttenuatorStatusEnum.HIGH)
        set_mock_value(second_bank.attenuators[4].position, AttenuatorStatusEnum.LOW)

        status = await second_bank.read()
        assert status == {
            "photon_energy": 12,
            "egu": "KeV",
            "total_transmission": pytest.approx(0.699),
            "attenuator_1": {"active": False, "transmission": 0},
            "attenuator_2": {
                "active": True,
                "transmission": pytest.approx(0.909, rel=0.001),
            },
            "attenuator_3": {
                "active": True,
                "transmission": pytest.approx(0.769, rel=0.001),
            },
            "attenuator_4": {"active": False, "transmission": 0},
        }

    def test_find_closest_attenuation(self, mock_attenuator_bank: AttenuatorBank):
        en = mock_attenuator_bank.energy.energy.readback.get()
        nearest = mock_attenuator_bank.find_closest_transmission(en, 0.7)
        assert nearest.transmission == 0.65

        nearest2 = mock_attenuator_bank.find_closest_transmission(en, 0.2)
        assert nearest2.transmission == 0.203

        nearest3 = mock_attenuator_bank.find_closest_transmission(en, 0.02)
        assert nearest3.transmission == 0.084

        nearest4 = mock_attenuator_bank.find_closest_transmission(en, 0.98)
        assert nearest4.transmission == 1

    def test_find_closest_attenuation_with_alt_energies(
        self, mock_attenuator_bank: AttenuatorBank
    ):
        en = mock_attenuator_bank.energy.energy.readback.get()
        nearest = mock_attenuator_bank.find_closest_transmission(en, 0.7)
        assert nearest == AttenuatorCombination(transmission=0.65, attenuators=[1, 2])
