from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, cast

from ophyd import Component as Cpt  # type: ignore[import-not-found]
from ophyd import (
    Device,
    EigerDetector,
    EpicsSignalRO,
    ProcessPlugin,
    ROIPlugin,
    StatsPlugin,
)
from ophyd.areadetector.base import (  # type: ignore[import-not-found]
    ADComponent,
    EpicsSignalWithRBV,
)
from ophyd.areadetector.filestore_mixins import (  # type: ignore[import-not-found]
    FileStoreBase,
    new_short_uid,
)
from ophyd.areadetector.trigger_mixins import (  # type: ignore[import-not-found]
    ADTriggerStatus,
    SingleTrigger,
)


class EigerFileHandler(Device, FileStoreBase):
    """A device to handle the file writing for the Eiger detector.

    When the Eiger's FileWriter module and SaveFiles are enabled, the file writing is handled
    by the detector itself. In this case, we want to generate a resource document for the
    file path and file name pattern. Then, we want to generate a datum for each trigger that
    enables us to get the individual frames from the file.

    The alternative to this is to use the Stream interface and configure the area detector plugins
    to write to a file store.
    """

    sequence_id = ADComponent(EpicsSignalRO, "SequenceId")
    file_path = ADComponent(EpicsSignalWithRBV, "FilePath", string=True)
    file_write_name_pattern = ADComponent(
        EpicsSignalWithRBV, "FWNamePattern", string=True
    )
    file_write_images_per_file = ADComponent(EpicsSignalWithRBV, "FWNImagesPerFile")
    enable = Cpt(EpicsSignalWithRBV, "FWEnable")
    data_source = Cpt(EpicsSignalWithRBV, "DataSource", string=True)
    save_files = Cpt(EpicsSignalWithRBV, "SaveFiles")

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        self.sequence_id_offset = 1
        super().__init__(*args, **kwargs)

        # NOTE: See `FileStoreBase._generate_resource` for the use of these.
        self._fn = None
        self.filestore_spec = "AD_EIGER"
        self._master_file_paths: list[PurePath] = []

    @property
    def master_file_paths(self) -> list[PurePath]:
        if len(self._master_file_paths) == 0:
            msg = "Master file path has not been set. Call stage() first."
            raise ValueError(msg)
        return self._master_file_paths

    @property
    def sequence_number(self) -> int:
        return self.sequence_id_offset + int(self.sequence_id.get())

    def stage(self) -> list[object]:  # type: ignore[reportIncompatibleMethodOverride]
        res_uid = new_short_uid()
        write_path = Path(f"{datetime.now().strftime(self.write_path_template)}/")
        self.file_path.set(write_path.as_posix()).wait(1.0)

        # The name pattern must have `$id` in it.
        # `$id` is replaced by the current sequence id of the acquisition.
        # E.g. * <res_uid>_1_master.h5
        #      * <res_uid>_1_data_000001.h5
        #      * <res_uid>_1_data_000002.h5
        #      * ...
        self.file_write_name_pattern.set(f"{res_uid}_$id").wait(1.0)

        ret: list[object] = super().stage()  # type: ignore[reportIncompatibleMethodOverride]

        # Set the filename for the resource document.
        file_prefix = PurePath(self.file_path.get()) / res_uid
        self._fn = file_prefix

        images_per_file: str = self.file_write_images_per_file.get()
        resource_kwargs: dict[str, str] = {"images_per_file": images_per_file}

        self._generate_resource(resource_kwargs)

        # Validate that the root path exists
        if not Path.exists(Path(self.reg_root)):
            msg = f"Root path {self.reg_root} does not exist"
            raise FileNotFoundError(msg)

        # Create the templated part of the path
        if not Path.exists(write_path):
            Path.mkdir(write_path, parents=True)

        # Reset the list of master file paths
        self._master_file_paths = []

        return ret

    def generate_datum(
        self, key: str, timestamp: float, datum_kwargs: dict[str, Any]
    ) -> Any:
        # The detector keeps its own counter which is uses label HDF5
        # sub-files.  We access that counter via the sequence_id
        # signal and stash it in the datum.
        datum_kwargs.update({"seq_id": self.sequence_number})
        self._master_file_paths.append(
            PurePath(f"{self._fn}_{self.sequence_number}_master.h5")
        )
        return super().generate_datum(key, timestamp, datum_kwargs)


class EigerBase(EigerDetector):
    """Base class for Eiger detectors that have the commonly used plugins."""

    file_handler = Cpt(
        EigerFileHandler,
        "cam1:",
        name="file_handler",
        # TODO: These paths need to be changed once the detector is deployed at CDI.
        write_path_template="/nsls2/data/tst/legacy/mock-proposals/2025-2/pass-56789/assets/eiger/%Y/%m/%d",
        root="/nsls2/data/tst/legacy/mock-proposals/2025-2/pass-56789/assets/eiger",
    )
    stats1 = Cpt(StatsPlugin, "Stats1:")
    stats2 = Cpt(StatsPlugin, "Stats2:")
    stats3 = Cpt(StatsPlugin, "Stats3:")
    stats4 = Cpt(StatsPlugin, "Stats4:")
    stats5 = Cpt(StatsPlugin, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    def stage(self, *args: Any, **kwargs: dict[str, Any]) -> list[object]:
        staged_devices: list[object] = super().stage(*args, **kwargs)
        self.cam.manual_trigger.set(True).wait(5.0)
        file_write_path: Path = Path(cast(str, self.file_handler.file_path.get()))
        if not Path.exists(file_write_path):
            msg = f"Path {file_write_path} does not exist."
            raise FileNotFoundError(msg)
        return staged_devices

    def unstage(self) -> list[object]:
        self.cam.manual_trigger.set(False).wait(5.0)
        ret = super().unstage()

        if not all(Path(path).exists() for path in self.file_handler.master_file_paths):
            msg = f"Paths {self.file_handler.master_file_paths} were not written."
            raise FileNotFoundError(msg)
        return ret


class EigerSingleTrigger(SingleTrigger, EigerBase):  # type: ignore[reportIncompatibleMethodOverride]
    """Eiger detector that uses the single trigger acquisition mode."""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.stage_sigs["cam.trigger_mode"] = 0
        self.stage_sigs["file_handler.data_source"] = "FileWriter"
        self.stage_sigs["file_handler.enable"] = True
        self.stage_sigs["file_handler.save_files"] = True

    def trigger(self, *args: Any, **kwargs: dict[str, Any]) -> ADTriggerStatus:
        status = super().trigger(*args, **kwargs)
        # If the manual trigger is enabled, we need to press the special trigger button
        # to actually trigger the detector.
        if self.cam.manual_trigger.get() == 1:
            self.cam.special_trigger_button.set(1).wait(5.0)
        return status
