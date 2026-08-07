import pytest
from dataclasses import dataclass
from bluesky.run_engine import RunEngine
from ophyd_async.core import init_devices, set_mock_value
from ophyd_async.epics.adcore import ADBaseDataType
from cditools.merlin_async import MerlinCounterDepth, MerlinDriverIO


@pytest.fixture
def mock_merlin_driver(RE: RunEngine) -> MerlinDriverIO:
    """Create a mock EigerDriverIO for testing."""
    with init_devices(mock=True):
        driver = MerlinDriverIO("MOCK:EIGER:cam1:")

    @dataclass
    class Parent:
        name: str

    driver.parent = Parent("merlin")

    # Set some mock values
    set_mock_value(driver.counter_depth, MerlinCounterDepth.BIT_12)

    return driver

async def test_driver_data_type(mock_merlin_driver: MerlinDriverIO):
    assert await mock_merlin_driver.data_type.get_value() == ADBaseDataType.UINT16

    set_mock_value(mock_merlin_driver.counter_depth, MerlinCounterDepth.BIT_24)
    assert await mock_merlin_driver.data_type.get_value() == ADBaseDataType.UINT32
