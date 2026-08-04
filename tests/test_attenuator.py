from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import bluesky.plans as bp
import pytest
import pytest_asyncio
from bluesky import RunEngine
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
        mock_energy = MagicMock(spec=Energy)
        mock_energy.energy.readback.get.return_value = photon_energy
        mock_energy.egu = "KeV"
        mock_attenuator = Attenuator(
            mock_attenuator_bank.prefix, 1, "Al", 16, mock_energy
        )
    yield mock_attenuator


class TestAttenuatorBankIntegration:
    @pytest.mark.asyncio
    async def test_count_plan(
        self, RE: RunEngine, mock_attenuator_bank: AttenuatorBank
    ):
        docs: dict[str, list[Any]] = {}

        def collect_doc(name: str, doc: dict[str, list[Any]]):
            docs.setdefault(name, []).append(doc)

        RE.subscribe(collect_doc)
        RE(bp.count([mock_attenuator_bank]))

        # --- Verify core document structure ---
        assert "start" in docs
        assert "stop" in docs
        assert len(docs["start"]) == 1
        assert len(docs["stop"]) == 1

        start_doc = docs["start"][0]
        assert start_doc["plan_name"] == "count"
        assert start_doc["num_points"] == 1
        assert start_doc["detectors"] == ["mock_attenuator_bank"]

        stop_doc = docs["stop"][0]
        assert stop_doc["exit_status"] == "success"
        assert stop_doc["run_start"] == start_doc["uid"]

        # --- Verify descriptor ---
        assert "descriptor" in docs
        descriptor = docs["descriptor"][0]
        assert "data_keys" in descriptor
        data_keys = descriptor["data_keys"]
        assert "mock_attenuator_bank-total_transmission" in data_keys
        assert "mock_attenuator_bank-attenuators-1_transmission" in data_keys
        assert "mock_attenuator_bank-attenuators-1_active" in data_keys
        assert data_keys["mock_attenuator_bank-total_transmission"] == {
            "source": "ca://test-prefix:total_transmission",
            "dtype": "number",
            "shape": [],
            "object_name": "mock_attenuator_bank",
        }

        # --- Verify event ---
        assert "event" in docs
        assert "data" in docs["event"][0]
        assert docs["event"][0]["data"] == {
            "mock_attenuator_bank-attenuators-1_transmission": 1.0,
            "mock_attenuator_bank-attenuators-1_active": False,
            "mock_attenuator_bank-attenuators-2_transmission": 1.0,
            "mock_attenuator_bank-attenuators-2_active": False,
            "mock_attenuator_bank-attenuators-3_transmission": 1.0,
            "mock_attenuator_bank-attenuators-3_active": False,
            "mock_attenuator_bank-attenuators-4_transmission": 1.0,
            "mock_attenuator_bank-attenuators-4_active": False,
            "mock_attenuator_bank-total_transmission": 1.0,
        }


class TestAttenuator:
    @pytest.mark.asyncio
    async def test_open(self, mock_attenuator: Attenuator):
        set_mock_value(mock_attenuator.position, AttenuatorStatusEnum.HIGH)
        await mock_attenuator.open()
        await assert_value(mock_attenuator.in_status, AttenuatorStatusEnum.LOW)

    @pytest.mark.asyncio
    async def test_close(self, mock_attenuator: Attenuator):
        set_mock_value(mock_attenuator.position, AttenuatorStatusEnum.LOW)
        await mock_attenuator.close()
        await assert_value(mock_attenuator.position, AttenuatorStatusEnum.HIGH)

    @pytest.mark.asyncio
    async def test_is_active(self, mock_attenuator: Attenuator):
        set_mock_value(mock_attenuator.position, AttenuatorStatusEnum.LOW)
        assert not await mock_attenuator.is_active()

        set_mock_value(mock_attenuator.position, AttenuatorStatusEnum.HIGH)
        assert await mock_attenuator.is_active()

    @pytest.mark.asyncio
    async def test_read(self, mock_attenuator: Attenuator):
        status = await mock_attenuator.read()
        expected_keys = ["mock_attenuator_active", "mock_attenuator_transmission"]
        assert all(key in status for key in expected_keys)

    @pytest.mark.asyncio
    async def test_describe(self, mock_attenuator: Attenuator):
        description = await mock_attenuator.describe()
        expected_keys = ["mock_attenuator_active", "mock_attenuator_transmission"]
        assert all(key in description for key in expected_keys)

    @pytest.mark.asyncio
    async def test_read_keys_match_describe_keys(self, mock_attenuator: Attenuator):
        status = await mock_attenuator.read()
        description = await mock_attenuator.describe()
        assert status.keys() == description.keys()

    def test_transmission_kev(self, mock_attenuator: Attenuator):
        assert mock_attenuator.transmission() == pytest.approx(0.84, abs=0.01)

    def test_transmission_ev(self):
        second_energy = MagicMock(spec=Energy)
        second_energy.energy.readback.get.return_value = 8600
        second_energy.egu = "eV"
        second_attenuator = Attenuator(prefix, 1, "Al", 16, second_energy)
        assert second_attenuator.transmission() == pytest.approx(0.84, abs=0.01)

    def test_attenuation_kev(self, mock_attenuator: Attenuator):
        assert mock_attenuator.attenuation() == pytest.approx(0.16, abs=0.01)

    def test_attenuation_ev(self):
        second_energy = MagicMock(spec=Energy)
        second_energy.energy.readback.get.return_value = 8600
        second_energy.egu = "eV"
        second_attenuator = Attenuator(prefix, 1, "Al", 16, second_energy)
        assert second_attenuator.attenuation() == pytest.approx(0.16, abs=0.01)


class TestAttenuatorBank:
    @pytest.mark.asyncio
    async def test_attenuation_bank_creation(
        self, mock_attenuator_bank: AttenuatorBank
    ):
        assert mock_attenuator_bank.energy.energy.readback.get() == 8.6
        assert mock_attenuator_bank.get_photon_energy() == 8.6

        second_energy = MagicMock(spec=Energy)
        second_energy.energy.readback.get.return_value = 6
        second_bank = AttenuatorBank(prefix, attenuator_configs, second_energy)
        assert second_bank.energy.energy.readback.get() == 6
        assert second_bank.get_photon_energy() == 6

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
        assert atten1.name == "mock_attenuator_bank-attenuators-1"

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
        expected_keys = {
            "mock_attenuator_bank-total_transmission",
            "mock_attenuator_bank-attenuators-1_active",
            "mock_attenuator_bank-attenuators-1_transmission",
            "mock_attenuator_bank-attenuators-2_active",
            "mock_attenuator_bank-attenuators-2_transmission",
            "mock_attenuator_bank-attenuators-3_active",
            "mock_attenuator_bank-attenuators-3_transmission",
            "mock_attenuator_bank-attenuators-4_active",
            "mock_attenuator_bank-attenuators-4_transmission",
        }
        assert all(key in status for key in expected_keys)

        # Test total transmission and attenuator values
        assert status["mock_attenuator_bank-total_transmission"]["value"] == 1.0
        assert not status["mock_attenuator_bank-attenuators-1_active"]["value"]
        assert status["mock_attenuator_bank-attenuators-1_transmission"]["value"] == 1.0
        assert not status["mock_attenuator_bank-attenuators-2_active"]["value"]
        assert status["mock_attenuator_bank-attenuators-2_transmission"]["value"] == 1.0
        assert not status["mock_attenuator_bank-attenuators-3_active"]["value"]
        assert status["mock_attenuator_bank-attenuators-3_transmission"]["value"] == 1.0
        assert not status["mock_attenuator_bank-attenuators-4_active"]["value"]
        assert status["mock_attenuator_bank-attenuators-4_transmission"]["value"] == 1.0

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

        expected_keys = {
            "second_bank-total_transmission",
            "second_bank-attenuators-1_active",
            "second_bank-attenuators-1_transmission",
            "second_bank-attenuators-2_active",
            "second_bank-attenuators-2_transmission",
            "second_bank-attenuators-3_active",
            "second_bank-attenuators-3_transmission",
            "second_bank-attenuators-4_active",
            "second_bank-attenuators-4_transmission",
        }

        status = await second_bank.read()
        assert all(key in status for key in expected_keys)

        # Test total transmission and attenuator values
        assert status["second_bank-total_transmission"]["value"] == pytest.approx(0.699)
        assert not status["second_bank-attenuators-1_active"]["value"]
        assert status["second_bank-attenuators-1_transmission"]["value"] == 1.0
        assert status["second_bank-attenuators-2_active"]["value"]
        assert status["second_bank-attenuators-2_transmission"][
            "value"
        ] == pytest.approx(0.909, rel=0.001)
        assert status["second_bank-attenuators-3_active"]["value"]
        assert status["second_bank-attenuators-3_transmission"][
            "value"
        ] == pytest.approx(0.769, rel=0.001)
        assert not status["second_bank-attenuators-4_active"]["value"]
        assert status["second_bank-attenuators-4_transmission"]["value"] == 1.0

    @pytest.mark.asyncio
    async def test_describe(self, mock_attenuator_bank: AttenuatorBank):
        description = await mock_attenuator_bank.describe()

        expected_keys = {
            "mock_attenuator_bank-total_transmission",
            "mock_attenuator_bank-attenuators-1_active",
            "mock_attenuator_bank-attenuators-1_transmission",
            "mock_attenuator_bank-attenuators-2_active",
            "mock_attenuator_bank-attenuators-2_transmission",
            "mock_attenuator_bank-attenuators-3_active",
            "mock_attenuator_bank-attenuators-3_transmission",
            "mock_attenuator_bank-attenuators-4_active",
            "mock_attenuator_bank-attenuators-4_transmission",
        }
        assert set(description.keys()) == expected_keys

        for i in range(1, 5):
            assert description[f"mock_attenuator_bank-attenuators-{i}_active"] == {
                "source": mock_attenuator_bank.attenuators[i].position.source,
                "dtype": "boolean",
                "shape": [],
            }
            assert description[
                f"mock_attenuator_bank-attenuators-{i}_transmission"
            ] == {
                "source": "method",
                "dtype": "number",
                "shape": [],
            }

        assert description["mock_attenuator_bank-total_transmission"] == {
            "source": f"ca://{mock_attenuator_bank.prefix}:total_transmission",
            "dtype": "number",
            "shape": [],
        }

    @pytest.mark.asyncio
    async def test_read_keys_match_describe_keys(
        self, mock_attenuator_bank: AttenuatorBank
    ):
        """This is expected behavior of the BlueskyInterface"""
        mock_attenuator_bank.set(1)
        read = await mock_attenuator_bank.read()
        description = await mock_attenuator_bank.describe()
        assert read.keys() == description.keys()

    def test_find_closest_attenuation(self, mock_attenuator_bank: AttenuatorBank):
        nearest = mock_attenuator_bank.find_closest_transmission(0.7)
        assert nearest.transmission == 0.65

        nearest2 = mock_attenuator_bank.find_closest_transmission(0.2)
        assert nearest2.transmission == 0.203

        nearest3 = mock_attenuator_bank.find_closest_transmission(0.02)
        assert nearest3.transmission == 0.084

        nearest4 = mock_attenuator_bank.find_closest_transmission(0.98)
        assert nearest4.transmission == 1

    def test_find_closest_attenuation_with_alt_energies(
        self, mock_attenuator_bank: AttenuatorBank
    ):
        nearest = mock_attenuator_bank.find_closest_transmission(0.7)
        assert nearest == AttenuatorCombination(transmission=0.65, attenuators=[1, 2])
