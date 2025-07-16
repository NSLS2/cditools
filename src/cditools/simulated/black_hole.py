from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable

from caproto import (  # type: ignore[import-not-found]
    ChannelChar,
    ChannelData,
    ChannelDouble,
    ChannelEnum,
    ChannelInteger,
    ChannelString,
)
from caproto.server import (  # type: ignore[import-not-found]
    PVGroup,
    ioc_arg_parser,
    run,
)

PLUGIN_TYPE_PVS = [
    (re.compile("image\\d:"), "NDPluginStdArrays"),
    (re.compile("Stats\\d:"), "NDPluginStats"),
    (re.compile("CC\\d:"), "NDPluginColorConvert"),
    (re.compile("Proc\\d:"), "NDPluginProcess"),
    (re.compile("Over\\d:"), "NDPluginOverlay"),
    (re.compile("ROI\\d:"), "NDPluginROI"),
    (re.compile("Trans\\d:"), "NDPluginTransform"),
    (re.compile("netCDF\\d:"), "NDFileNetCDF"),
    (re.compile("TIFF\\d:"), "NDFileTIFF"),
    (re.compile("JPEG\\d:"), "NDFileJPEG"),
    (re.compile("Nexus\\d:"), "NDPluginNexus"),
    (re.compile("HDF\\d:"), "NDFileHDF5"),
    (re.compile("Magick\\d:"), "NDFileMagick"),
    (re.compile("TIFF\\d:"), "NDFileTIFF"),
    (re.compile("HDF\\d:"), "NDFileHDF5"),
    (re.compile("Current\\d:"), "NDPluginStats"),
    (re.compile("SumAll"), "NDPluginStats"),
]


class ReallyDefaultDict(defaultdict[str, ChannelData]):
    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.default_factory: Callable[[str], ChannelData] = self.default_factory  # type: ignore[assignment]

    def __contains__(self, key: object) -> bool:
        return True

    def __missing__(self, key: str) -> ChannelData:
        if key.endswith(("-SP", "-I", "-RB", "-Cmd")):
            key, *_ = key.rpartition("-")
            return self[key]
        if key.endswith(("_RBV", ":RBV")):
            return self[key[:-4]]
        self[key] = self.default_factory(key)
        return self[key]


class CDIBlackHoleIOC(PVGroup):
    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        # Initialize the explicit pv properties
        super().__init__(*args, prefix="", **kwargs)
        # Overwrite the pvdb with the blackhole, while keeping the explicit pv properties
        self.pvdb: dict[str, ChannelData] = self.pvdb
        self.old_pvdb = self.pvdb.copy()
        self.pvdb = ReallyDefaultDict(self.fabricate_channel)  # type: ignore[reportIncompatibleMethodOverride]

    def fabricate_channel(self, key: str) -> ChannelData:
        # If the channel already exists from initialization, return it
        if key in self.old_pvdb:
            return self.old_pvdb[key]
        # Otherwise, fabricate new channels
        if "PluginType" in key:
            for pattern, val in PLUGIN_TYPE_PVS:
                if pattern.search(key):
                    return ChannelString(value=val)
        elif "ArrayPort" in key:
            return ChannelString(value="cam1")
        elif "PortName" in key:
            # Extract port name from key format: <prefix><port-name>:PortName
            # Use regex to find the last component before :PortName
            match = re.search(r"[^:}_\-]+(?=:PortName)", key)
            if match:
                port_name = match.group(0)
                return ChannelString(value=port_name)
            # Fallback if regex doesn't match
            return ChannelString(value=key)
        elif "name" in key.lower():
            return ChannelString(value=key)
        elif "EnableCallbacks" in key:
            return ChannelEnum(value=0, enum_strings=["Disabled", "Enabled"])
        elif "BlockingCallbacks" in key or "Auto" in key:
            return ChannelEnum(value=0, enum_strings=["No", "Yes"])
        elif "ImageMode" in key:
            return ChannelEnum(
                value=0, enum_strings=["Single", "Multiple", "Continuous"]
            )
        elif "WriteMode" in key:
            return ChannelEnum(value=0, enum_strings=["Single", "Capture", "Stream"])
        elif "ArraySize" in key:
            return ChannelData(value=10)
        elif "TriggerMode" in key:
            return ChannelEnum(value=0, enum_strings=["Internal", "External"])
        elif "FileWriteMode" in key:
            return ChannelEnum(value=0, enum_strings=["Single"])
        elif "FilePathExists" in key:
            return ChannelData(value=1)
        elif "WaitForPlugins" in key:
            return ChannelEnum(value=0, enum_strings=["No", "Yes"])
        elif (
            "file" in key.lower()
            and "number" not in key.lower()
            and "mode" not in key.lower()
        ):
            return ChannelChar(value="a" * 250)
        elif "filenumber" in key.lower():
            return ChannelInteger(value=0)
        elif "Compression" in key:
            return ChannelEnum(
                value=0, enum_strings=["None", "N-bit", "szip", "zlib", "blosc"]
            )
        return ChannelDouble(value=0.0)


def main() -> None:
    _, run_options = ioc_arg_parser(default_prefix="", desc="PV black hole")
    run_options["interfaces"] = ["127.0.0.1"]
    run(CDIBlackHoleIOC().pvdb, **run_options)
