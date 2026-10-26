from __future__ import annotations
from collections import OrderedDict

import numpy as np
import time as ttime
from ophyd import (
    CamBase,
    DerivedSignal,
    Device,
    EpicsMotor,
    EpicsSignal,
    ProsilicaDetector,
    ProsilicaDetectorCam,
    ROIPlugin,
)
from ophyd import Component as Cpt
from ophyd.areadetector.plugins import (
    PluginBase,
    ROIStatNPlugin_V25,
    ROIStatPlugin_V35,
    StatsPlugin,
    TransformPlugin,
    HDF5Plugin_V34 as HDF5Plugin
)
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.device import DynamicDeviceComponent


class FullROIStats(ROIStatPlugin_V35):
    rois = DynamicDeviceComponent(
        {f"roi{j}": (ROIStatNPlugin_V25, f"{j}:", {}) for j in range(1, 9)}
    )

    def set_from_epics(self):
        root = self
        while root.parent is not None:
            root = root.parent

        for k in self.rois.component_names:
            roi = getattr(self.rois, k)
            in_use = roi.use.get()
            if in_use == "Yes":
                roi.kind = "normal"
            else:
                roi.kind = "omitted"
                continue
            epics_name = roi.name_.get()
            roi.name = f"{root.name}_{epics_name}"
            for cpt in roi.walk_signals():
                cpt.item.name = f"{root.name}_{epics_name}_{cpt.dotted_name}"


class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    """Add this as a component to detectors that write HDF5s."""

    def warmup(self):
        """
        This is vendored from ophyd (https://github.com/bluesky/ophyd/blob/master/ophyd/areadetector/plugins.py)
        to fix the non-existent "Internal" trigger mode that is hard-coded there:

            In [13]: cam6.stage()
            An exception has occurred, use '%tb verbose' to see the full traceback.
            UnprimedPlugin: The plugin hdf5 on the area detector with name cam6 has not been primed.

            See /home/xf08bm/bluesky-files/log/bluesky/bluesky.log for the full traceback.

            In [14]: cam6.hdf5.warmup()
            An exception has occurred, use '%tb verbose' to see the full traceback.
            ValueError: invalid literal for int() with base 0: b'Internal'

            See /home/xf08bm/bluesky-files/log/bluesky/bluesky.log for the full traceback.
        """
        self.enable.set(1).wait()
        sigs = OrderedDict([(self.parent.cam.array_callbacks, 1),
                            (self.parent.cam.image_mode, 'Single'),
                            (self.parent.cam.trigger_mode, 'Fixed Rate'),  # updated here "Internal" -> "Fixed Rate"
                            # just in case tha acquisition time is set very long...
                            (self.parent.cam.acquire_time, 1),
                            (self.parent.cam.acquire_period, 1),
                            (self.parent.cam.acquire, 1)])

        original_vals = {sig: sig.get() for sig in sigs}

        for sig, val in sigs.items():
            ttime.sleep(0.1)  # abundance of caution
            sig.set(val).wait()

        ttime.sleep(2)  # wait for acquisition

        for sig, val in reversed(list(original_vals.items())):
            ttime.sleep(0.1)
            sig.set(val).wait()


class ProsilicaCamBase(ProsilicaDetector):
    hdf5 = Cpt(HDF5PluginWithFileStore,
               suffix="HDF1:",
               write_path_template="/tmp",
               root="/nsls2/data/cdi/proposals")
    wait_for_plugins = Cpt(EpicsSignal, "WaitForPlugins", string=True, kind="hinted")
    cam = Cpt(ProsilicaDetectorCam, "cam1:")
    stats1 = Cpt(StatsPlugin, "Stats1:")
    stats2 = Cpt(StatsPlugin, "Stats2:")
    stats3 = Cpt(StatsPlugin, "Stats3:")
    stats4 = Cpt(StatsPlugin, "Stats4:")
    stats5 = Cpt(StatsPlugin, "Stats5:")
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    roistat1 = Cpt(FullROIStats, "ROIStat1:")
    _default_plugin_graph: dict[PluginBase, CamBase | PluginBase] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roistat1.kind = "hinted"
        self._use_default_plugin_graph: bool = True

    @property
    def default_plugin_graph(
        self,
    ) -> dict[PluginBase, CamBase | PluginBase] | None:
        return self._default_plugin_graph

    @property
    def root_path_str(self):
        return f"{self.hdf5.root}/{self._md['cycle']}/{self._md['data_session']}/assets/{self._asset_path}"

    @property
    def _asset_path(self):
        # cam names are 'cam_{hutch}{num}'
        # need to split, slice off hutch, and join on a dash for directory
        base, num = self.name.split("_")[0:2]
        num = num[1:]
        return f"{base}-{num}"

    def _stage_plugin_graph(self, plugin_graph: dict[PluginBase, CamBase | PluginBase]):
        for target, source in plugin_graph.items():
            self.stage_sigs[target.nd_array_port] = source.port_name.get()
            self.stage_sigs[target.enable] = True

    def stage(self):
        if self._use_default_plugin_graph and self.default_plugin_graph is not None:
            self._stage_plugin_graph(self.default_plugin_graph)

        return super().stage()


class StandardProsilicaCam(SingleTrigger, ProsilicaCamBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_plugin_graph = {
            self.stats1: self.roi1,
            self.stats2: self.roi2,
            self.stats3: self.roi3,
            self.stats4: self.roi4,
            self.stats5: self.cam,
            self.trans1: self.cam,
            self.roi1: self.cam,
            self.roi2: self.cam,
            self.roi3: self.cam,
            self.roi4: self.cam,
            self.roistat1: self.cam,
        }

class ScreenState(DerivedSignal):
    def __init__(self, *args, in_position=0.0, out_position=25.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_position = in_position
        self.out_position = out_position

    def forward(self, value):
        msg = "Forward method is not implemented."
        raise NotImplementedError(msg)

    def inverse(self, value):
        if np.isclose(value, self.in_position, atol=1):
            return "in"
        if value > self.out_position:
            return "out"
        return "invalid"


class StandardScreen(Device):
    mtr = Cpt(EpicsMotor, "")
    state = Cpt(
        ScreenState,
        derived_from="mtr.user_readback",
        in_position=0.0,
        out_position=25.0,
    )

    def set(
        self,
        new_position,
        *,
        timeout: float | None = None,
        moved_cb=None,
        wait: bool | None = None,
    ):
        if new_position == "in":
            return self.mtr.set(
                self.state.in_position, timeout=timeout, moved_cb=moved_cb, wait=wait
            )
        if new_position == "out":
            return self.mtr.set(
                self.state.out_position, timeout=timeout, moved_cb=moved_cb, wait=wait
            )
        msg = f"Invalid position '{new_position}'. Use 'in' or 'out'."
        raise ValueError(msg)


def set_roiN_kinds(cam):
    cam.roistat1.rois.kind = "normal"
    for roi_name in cam.roistat1.rois.component_names:
        roi = getattr(cam.roistat1.rois, roi_name)
        roi.kind = "normal"
        for k in ("bgd_width", "name_", "use", "size", "min_"):
            getattr(roi, k).kind = "config"
        for k in ("max_value", "min_value", "mean_value", "net"):
            getattr(roi, k).kind = "normal"
        for k in ("total",):
            getattr(roi, k).kind = "hinted"
        for k in ("reset",):
            getattr(roi, k).kind = "omitted"
        for k in roi.component_names:
            if k.startswith("ts"):
                getattr(roi, k).kind = "omitted"
    cam.roistat1.set_from_epics()
    return cam
