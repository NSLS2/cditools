from __future__ import annotations

import logging
from collections import OrderedDict

from ophyd import (
    AreaDetector,
    CamBase,
    Device,
    EpicsSignal,
    HDF5Plugin,
    ProcessPlugin,
    ROIPlugin,
    StatsPlugin,
    TIFFPlugin,
    TransformPlugin,
)
from ophyd import Component as Cpt
from ophyd.areadetector import EpicsSignalWithRBV
from ophyd.areadetector.base import ADComponent
from ophyd.areadetector.filestore_mixins import FileStorePluginBase, FileStoreTIFF
from ophyd.utils.paths import makedirs

from .trigger_mixins import CDIModalTrigger, FileStoreBulkReadable

logger = logging.getLogger(__name__)


class MerlinTiffPlugin(TIFFPlugin, FileStoreBulkReadable, FileStoreTIFF, Device):
    def mode_external(self) -> None:
        total_points = self.parent.mode_settings.total_points.get()  # type: ignore[union-attr]
        self.stage_sigs[self.num_capture] = total_points

    def get_frames_per_point(self) -> object:
        mode = self.parent.mode_settings.mode.get()  # type: ignore[union-attr]
        if mode == "external":
            return 1
        return self.parent.cam.num_images.get()  # type: ignore[union-attr]

    def describe(self) -> object:
        ret = super().describe()
        key = self.parent._image_name  # type: ignore[union-attr]
        ret[key].setdefault("dtype_str", "<u2")  # type: ignore[attr-defined]
        return ret


class MerlinDetectorCam(CamBase):
    acquire = ADComponent(EpicsSignal, "Acquire")
    quad_merlin_mode = ADComponent(EpicsSignalWithRBV, "QuadMerlinMode")


class MerlinDetector(AreaDetector):
    cam = Cpt(
        MerlinDetectorCam,
        "cam1:",
        read_attrs=[],
        configuration_attrs=[
            "image_mode",
            "trigger_mode",
            "acquire_time",
            "acquire_period",
        ],
    )


class MerlinFileStoreHDF5(FileStorePluginBase, FileStoreBulkReadable):
    _spec = "TPX_HDF5"
    filestore_spec = _spec

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage_sigs.update(
            [
                (self.file_template, "%s%s_%6.6d.h5"),  # type: ignore[attr-defined]
                (self.file_write_mode, "Stream"),  # type: ignore[attr-defined]
                (self.compression, "zlib"),  # type: ignore[attr-defined]
                (self.capture, 1),  # type: ignore[attr-defined]
            ]
        )

    def stage(self) -> object:
        logger.info("Staging")
        staged = super().stage()
        logger.info("Staging step 2")
        res_kwargs = {"frame_per_point": 1}
        logger.info("res_kwargs = {frame_per_point: %s}", res_kwargs["frame_per_point"])

        logger.debug("Inserting resource with filename %s", self._fn)
        logger.info("Inserting resource with filename %s", self._fn)
        self._generate_resource(res_kwargs)
        logger.info("generating resources")
        logger.info("Staged")
        return staged

    def describe(self) -> OrderedDict[str, dict]:
        ret = super().describe()
        key = self.parent._image_name  # type: ignore[union-attr]
        ret[key].setdefault("dtype_str", "<u2")  # type: ignore[attr-defined]
        return ret  # type: ignore[return-value]

    def make_filename(self) -> tuple[str, str, str]:
        fn, read_path, write_path = super().make_filename()
        mode_settings = self.parent.mode_settings  # type: ignore[union-attr]
        if mode_settings.make_directories.get():
            makedirs(read_path)
        return fn, read_path, write_path


class HDF5PluginWithFileStore(HDF5Plugin, MerlinFileStoreHDF5):
    def stage(self) -> object:
        mode_settings = self.parent.mode_settings  # type: ignore[union-attr]
        total_points = mode_settings.total_points.get()
        self.stage_sigs[self.num_capture] = total_points

        # ensure that setting capture is the last thing that's done
        self.stage_sigs.move_to_end(self.capture)
        return super().stage()

    def describe(self):
        ret = super().describe()
        key = self.parent._image_name  # type: ignore[union-attr]
        ret[key].setdefault("dtype_str", "<u2")  # type: ignore[attr-defined]
        return ret


class CDIMerlinDetector(CDIModalTrigger, MerlinDetector):
    hdf5 = Cpt(
        HDF5PluginWithFileStore,
        "HDF1:",
        read_attrs=[],
        configuration_attrs=[],
        write_path_template="/nsls2/data/tst/legacy/mock-proposals/2025-2/pass-56789/assets/merlin/%Y/%m/%d",
        root="/nsls2/data/tst/legacy/mock-proposals/2025-2/pass-56789/assets/merlin",
    )

    proc1 = Cpt(ProcessPlugin, "Proc1:")
    stats1 = Cpt(StatsPlugin, "Stats1:")
    stats2 = Cpt(StatsPlugin, "Stats2:")
    stats3 = Cpt(StatsPlugin, "Stats3:")
    stats4 = Cpt(StatsPlugin, "Stats4:")
    stats5 = Cpt(StatsPlugin, "Stats5:")
    transform1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")

    def __init__(
        self,
        prefix,
        *,
        read_attrs: list[str] | None = None,
        configuration_attrs: list[str] | None = None,
        **kwargs,
    ):
        if read_attrs is None:
            read_attrs = ["hdf5", "cam"]
        if configuration_attrs is None:
            configuration_attrs = ["hdf5", "cam"]

        if "hdf5" not in read_attrs:
            # ensure that hdf5 is still added, or data acquisition will fail
            read_attrs = [*list(read_attrs), "hdf5"]

        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            **kwargs,
        )

    def mode_internal(self) -> None:
        super().mode_internal()

        count_time = self.count_time.get()
        if isinstance(count_time, float):
            self.stage_sigs[self.cam.acquire_time] = count_time
            self.stage_sigs[self.cam.acquire_period] = count_time + 0.005

    def mode_external(self) -> None:
        super().mode_external()

        # NOTE: these values specify a debounce time for external triggering so
        #       they should be set to < 0.5 the expected exposure time, or at
        #       minimum the lowest possible dead time = 1.64ms
        expected_exposure = 0.001
        min_dead_time = 0.00164
        self.stage_sigs[self.cam.acquire_time] = expected_exposure
        self.stage_sigs[self.cam.acquire_period] = expected_exposure + min_dead_time

        self.cam.stage_sigs[self.cam.trigger_mode] = "Trigger Enable"
