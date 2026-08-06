from dataclasses import dataclass

from bluesky.run_engine import RunEngine
from ophyd_async.core import init_devices
import pytest

from cditools.eiger_async import EigerDriverIO
from ophyd_async.epics.adcore import ADBaseDataType

from cditools.merlin_async import MerlinCounterDepth


@pytest.fixture
def mock_merlin_driver(RE: RunEngine) -> EigerDriverIO:
    """Create a mock EigerDriverIO for testing."""
    with init_devices(mock=True):
        driver = EigerDriverIO("MOCK:EIGER:cam1:")

    @dataclass
    class Parent:
        name: str

    driver.parent = Parent("merlin")

    # Set some mock values
    driver.counter_depth = MerlinCounterDepth.BIT_12

    return driver

def test_driver_data_type(mock_merlin_driver: EigerDriverIO):
    assert mock_merlin_driver.data_type == ADBaseDataType.UINT16

    mock_merlin_driver.counter_depth = MerlinCounterDepth.BIT_24
    assert mock_merlin_driver.data_type == ADBaseDataType.UINT32
