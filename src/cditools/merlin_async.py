"""
Ophyd Async implementation for the Merlin Detector
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated as A

from ophyd_async.core import (
    DetectorAcquireLogic,
    DetectorTriggerLogic,
    SignalDict,
    SignalR,
    SignalRW,
    StrictEnum,
    SubsetEnum,
    soft_signal_rw,
)
from ophyd_async.epics.adcore import (
    ADBaseDataType,
    ADBaseIO,
    ADWriterFactory,
    AreaDetector,
    NDPluginBaseIO,
    prepare_exposures,
    trigger_info_from_num_images,
)
from ophyd_async.epics.core import PvSuffix

__all__ = [
    "MerlinDetector",
    "MerlinDriverIO",
    "MerlinTriggerLogic",
]

_MIN_DEAD_TIME = 0.002


class MerlinTriggerMode(SubsetEnum):
    """Trigger modes for the Merlin detector"""

    INTERNAL = "Internal"
    TRIGGER_ENABLE = "Trigger Enable"
    TRIGGER_START_RISING = "Trigger start rising"
    TRIGGER_START_FALLING = "Trigger start falling"
    TRIGGER_BOTH_RISING = "Trigger both rising "
    SOFTWARE = "Software"


class MerlinTriggerModeRBV(StrictEnum):
    """Trigger modes for the Merlin detector"""

    INTERNAL = "Internal"
    TRIGGER_ENABLE = "Trigger Enable"
    TRIGGER_START_RISING = "Trigger start rising"
    TRIGGER_START_FALLING = "Trigger start falling"
    TRIGGER_BOTH_RISING = "Trigger both rising "
    SOFTWARE = "Software"


class MerlinDriverIO(ADBaseIO):
    """Driver for merlin model:DU897_BV as deployed on p99.

    This mirrors the interface provided by ADMerlin/db/merlin.template.
    https://github.com/areaDetector/ADMerlin/blob/master/merlinApp/Db/merlin.template
    """

    trigger_mode: A[SignalRW[MerlinTriggerMode], PvSuffix.rbv("TriggerMode")]

    # Since ADMerlin doesn't set the data type readback correctly, but is always uint16,
    # just turn it into a static soft signal
    def __init__(self, prefix: str, name: str = ""):
        super().__init__(prefix, name=name)
        self.data_type = soft_signal_rw(
            ADBaseDataType, ADBaseDataType.UINT16, name="data_type"
        )


# The deadtime of an Merlin controller varies depending on the exact model of camera.
# Ideally we would maximize performance by dynamically retrieving the deadtime at
# runtime. See https://github.com/bluesky/ophyd-async/issues/308
@dataclass
class MerlinTriggerLogic(DetectorTriggerLogic):
    """Trigger logic for MerlinDriverIO."""

    driver: MerlinDriverIO

    def get_deadtime(self, config_values: SignalDict) -> float:  # noqa: ARG002
        return _MIN_DEAD_TIME

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        await self.driver.trigger_mode.set(MerlinTriggerMode.INTERNAL)
        await prepare_exposures(self.driver, num, livetime, deadtime)

    async def prepare_edge(self, num: int, livetime: float):
        await self.driver.trigger_mode.set(MerlinTriggerMode.TRIGGER_START_RISING)
        await prepare_exposures(self.driver, num, livetime)

    async def default_trigger_info(self):
        return await trigger_info_from_num_images(self.driver)


class MerlinDetector(AreaDetector[MerlinDriverIO]):
    """Create an ADMerlin AreaDetector instance.

    :param prefix: EPICS PV prefix for the detector
    :param path_provider: Provider for file paths during acquisition
    :param driver_suffix: Suffix for the driver PV, defaults to "cam1:"
    :param writer_type: Type of file writer (HDF or TIFF)
    :param writer_suffix: Suffix for the writer PV
    :param plugins: Additional areaDetector plugins to include
    :param config_sigs: Additional signals to include in configuration
    :param name: Name for the detector device
    """

    def __init__(
        self,
        prefix: str,
        *writer_factories: ADWriterFactory,
        driver_suffix="cam1:",
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = MerlinDriverIO(prefix + driver_suffix)
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=DetectorAcquireLogic(driver),
            trigger_logic=MerlinTriggerLogic(driver),
            plugins=plugins,
            config_sigs=config_sigs,
            name=name,
        )
