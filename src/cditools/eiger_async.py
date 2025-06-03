"""
Ophyd Async implementation for Eiger detector.
"""
from typing import Sequence, Annotated as A

from ophyd_async.core import PathProvider, TriggerInfo, SignalRW, SignalR, DetectorTrigger
from ophyd_async.epics.signal import PvSuffix
from ophyd_async.epics.adcore import AreaDetector, ADWriter, NDPluginBaseIO, ADBaseController, ADBaseIO


class EigerTriggerInfo(TriggerInfo):
    photon_energy: float


class EigerWriter(ADWriter):
    pass


class EigerDriverIO(ADBaseIO):
    photon_energy: A[SignalRW[float], PvSuffix.rbv("PhotonEnergy")]
    trigger_mode: A[SignalRW[str], PvSuffix.rbv("TriggerMode")]


class EigerController(ADBaseController[EigerDriverIO]):
    def __init__(self, driver: EigerDriverIO):
        self._driver = driver
        super().__init__(driver)

    async def prepare(self, trigger_info: EigerTriggerInfo):
        await self._driver.photon_energy.set(trigger_info.photon_energy)

        if trigger_info.trigger != DetectorTrigger.INTERNAL:
            raise NotImplementedError("Only internal trigger is supported")
        else:
            await self._driver.trigger_mode.set("ints")

        await super().prepare(trigger_info)


class EigerDetector(AreaDetector[EigerController]):

    def __init__(
        self,
        prefix: str,
        path_provider: PathProvider,
        driver_suffix: str = "cam1:",
        writer_cls: type[ADWriter] = EigerWriter,
        fileio_suffix: str = "cam1:",
        name: str = "",
        config_sigs: Sequence[SignalR] = (),
        plugins: dict[str, NDPluginBaseIO] = {},
    ):
        driver = EigerDriverIO(prefix + driver_suffix)
        controller = EigerController(driver)
        writer = writer_cls.with_io(
            prefix,
            path_provider,
            dataset_source=driver,
            fileio_suffix=fileio_suffix,
            plugins=plugins,
        )
        super().__init__(
            controller=controller,
            writer=writer,
            plugins=plugins,
            name=name,
            config_sigs=config_sigs,
        )
