from __future__ import annotations

import pytest
import pytest_asyncio
from ophyd_async.core import get_mock_put, init_devices, set_mock_value
from ophyd_async.testing import assert_value

from cditools.attenuator import (
    AVAILABLE_ATTENUATIONS,
    Attenuator,
    AttenuatorBank,
    AttenuatorStatusEnum,
)

pytest_plugins = ("pytest_asyncio",)
photon_energy = 8.6  # KeV


@pytest_asyncio.fixture
async def mock_attenuator_bank():
    async with init_devices(mock=True):
        mock_attenuator_bank = AttenuatorBank()
    yield mock_attenuator_bank


@pytest_asyncio.fixture
async def mock_attenuator(mock_attenuator_bank: AttenuatorBank):
    async with init_devices(mock=True):
        mock_attenuator = Attenuator(mock_attenuator_bank.prefix, 1, 16)
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
        assert mock_attenuator.transmission(photon_energy, units="eV") == pytest.approx(
            0.84, abs=0.01
        )

    def test_attenuation_kev(self, mock_attenuator: Attenuator):
        assert mock_attenuator.attenuation(photon_energy) == pytest.approx(
            0.16, abs=0.01
        )

    def test_attenuation_ev(self, mock_attenuator: Attenuator):
        photon_energy = 8600  # eV
        assert mock_attenuator.attenuation(photon_energy, units="eV") == pytest.approx(
            0.16, abs=0.01
        )


class TestAttenuatorBank:
    @pytest.mark.asyncio
    async def test_attenuators_indexed_at_1(self, mock_attenuator_bank: AttenuatorBank):
        with pytest.raises(KeyError):
            mock_attenuator_bank.attenuators[0]

        atten1 = mock_attenuator_bank.attenuators[1]
        assert atten1.num == 1
        assert atten1.thickness == 16
        assert atten1.position.source == "mock+ca://XF:09ID1-ES{IOLOGIK1:E1212}:DO1-Sts"
        assert atten1.mode.source == "mock+ca://XF:09ID1-ES{IOLOGIK1:E1212}:DIO1-Mode"
        assert (
            atten1.in_status.source == "mock+ca://XF:09ID1-ES{IOLOGIK1:E1212}:DI1-Sts"
        )

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
    async def test_set_attenuators(self, mock_attenuator_bank: AttenuatorBank):
        atten_mock1 = get_mock_put(mock_attenuator_bank.attenuators[1].position)
        atten_mock2 = get_mock_put(mock_attenuator_bank.attenuators[2].position)
        atten_mock3 = get_mock_put(mock_attenuator_bank.attenuators[3].position)
        atten_mock4 = get_mock_put(mock_attenuator_bank.attenuators[4].position)

        # AttenuatorCombination(attenuation=0.095, attenuators=[1, 2, 3]),
        combo0 = AVAILABLE_ATTENUATIONS[1]  # attenuators 1,2,3
        await mock_attenuator_bank.set(combo0.transmission)
        atten_mock1.assert_called_with(AttenuatorStatusEnum.LOW)
        atten_mock2.assert_called_with(AttenuatorStatusEnum.HIGH)
        atten_mock3.assert_called_with(AttenuatorStatusEnum.HIGH)
        atten_mock4.assert_called_with(AttenuatorStatusEnum.HIGH)

        # AttenuatorCombination(attenuation=0.768, attenuators=[1]),
        combo1 = AVAILABLE_ATTENUATIONS[-3]
        await mock_attenuator_bank.set(combo1.transmission)
        atten_mock1.assert_called_with(AttenuatorStatusEnum.LOW)
        atten_mock2.assert_called_with(AttenuatorStatusEnum.HIGH)
        atten_mock3.assert_called_with(AttenuatorStatusEnum.LOW)
        atten_mock4.assert_called_with(AttenuatorStatusEnum.LOW)

    @pytest.mark.asyncio
    async def test_get_bank_status(self, mock_attenuator_bank: AttenuatorBank):
        set_mock_value(
            mock_attenuator_bank.attenuators[1].position, AttenuatorStatusEnum.LOW
        )
        set_mock_value(
            mock_attenuator_bank.attenuators[2].position, AttenuatorStatusEnum.LOW
        )
        set_mock_value(
            mock_attenuator_bank.attenuators[3].position, AttenuatorStatusEnum.HIGH
        )
        set_mock_value(
            mock_attenuator_bank.attenuators[4].position, AttenuatorStatusEnum.LOW
        )

        assert await mock_attenuator_bank.get_status() == [
            AttenuatorStatusEnum.LOW,
            AttenuatorStatusEnum.LOW,
            AttenuatorStatusEnum.HIGH,
            AttenuatorStatusEnum.LOW,
        ]

    def test_find_closest_attenuation(self, mock_attenuator_bank: AttenuatorBank):
        nearest = mock_attenuator_bank.find_closest_attenuation(0.7)
        assert nearest.transmission == 0.65

        nearest2 = mock_attenuator_bank.find_closest_attenuation(0.2)
        assert nearest2.transmission == 0.203

        nearest3 = mock_attenuator_bank.find_closest_attenuation(0.02)
        assert nearest3.transmission == 0.084

        nearest4 = mock_attenuator_bank.find_closest_attenuation(0.98)
        assert nearest4.transmission == 1

    def test_up_to_date_available_attenuations(
        self, mock_attenuator_bank: AttenuatorBank
    ):
        assert (
            mock_attenuator_bank._calculate_available_attentuations(photon_energy)  # type: ignore[reportPrivateUsage]
            == AVAILABLE_ATTENUATIONS
        )
