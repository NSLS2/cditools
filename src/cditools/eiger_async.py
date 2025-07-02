"""
Ophyd Async implementation for Eiger detector.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Annotated as A
from typing import Any

import numpy as np
from bluesky.protocols import StreamAsset
from event_model import DataKey
from ophyd_async.core import (
    DatasetDescriber,
    DetectorTrigger,
    DetectorWriter,
    HDFDatasetDescription,
    HDFDocumentComposer,
    PathInfo,
    PathProvider,
    SignalR,
    SignalRW,
    StrictEnum,
    TriggerInfo,
    observe_value,
)
from ophyd_async.epics.adcore import (
    ADBaseController,
    ADBaseIO,
    ADImageMode,
    ADWriter,
    AreaDetector,
    NDPluginBaseIO,
)
from ophyd_async.epics.signal import PvSuffix

logger = getLogger(__name__)


class EigerTriggerMode(StrictEnum):
    """Trigger modes for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#implementation-of-standard-driver-parameters
    """

    INTERNAL_SERIES = "ints"
    INTERNAL_ENABLE = "int"
    EXTERNAL_SERIES = "exts"
    EXTERNAL_ENABLE = "ext"
    EXTERNAL_GATE = "gate"


class EigerExtGateMode(StrictEnum):
    """External gate modes for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#trigger-setup
    """

    PUMP_AND_PROBE = "Pump & Probe"
    HDR = "HDR"


class EigerROIMode(StrictEnum):
    """ROI modes for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#readout-setup
    """

    DISABLED = "Disabled"
    _4M = "4M"


class EigerCompressionAlgo(StrictEnum):
    """Compression algorithms for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#readout-setup
    """

    LZ4 = "lz4"
    BSLZ4 = "bslz4"
    NONE = "None"


class EigerDataSource(StrictEnum):
    """Data sources for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#readout-setup
    """

    NONE = "None"
    FILE_WRITER = "FileWriter"
    STREAM = "Stream"


class EigerHDF5Format(StrictEnum):
    """HDF5 formats for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#filewriter-interface
    """

    LEGACY = "Legacy"
    V2024_2 = "v2024.2"


class EigerStreamVersion(StrictEnum):
    """Stream versions for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#stream-interface
    """

    STREAM1 = "Stream1"
    STREAM2 = "Stream2"


class EigerStreamHdrDetail(StrictEnum):
    """Header detail levels for the Eiger detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#stream-interface
    """

    ALL = "All"
    BASIC = "Basic"
    NONE = "None"


class EigerDriverIO(ADBaseIO):
    """Defines the full specifics of the Eiger driver.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#implementation-of-standard-driver-parameters
    """

    # Standard Driver Parameters
    trigger_mode: A[SignalRW[EigerTriggerMode], PvSuffix.rbv("TriggerMode")]
    num_images: A[SignalRW[int], PvSuffix.rbv("NumImages")]
    num_images_counter: A[SignalR[int], PvSuffix("NumImagesCounter_RBV")]
    num_exposures: A[SignalRW[int], PvSuffix.rbv("NumExposures")]
    acquire_time: A[SignalRW[float], PvSuffix.rbv("AcquireTime")]
    acquire_period: A[SignalRW[float], PvSuffix.rbv("AcquirePeriod")]
    data_type: A[SignalRW[str], PvSuffix.rbv("DataType")]
    temperature_actual: A[SignalRW[float], PvSuffix.rbv("TemperatureActual")]
    max_size_x: A[SignalRW[int], PvSuffix.rbv("MaxSizeX")]
    max_size_y: A[SignalRW[int], PvSuffix.rbv("MaxSizeY")]
    array_size_x: A[SignalRW[int], PvSuffix.rbv("ArraySizeX")]
    array_size_y: A[SignalRW[int], PvSuffix.rbv("ArraySizeY")]
    manufacturer: A[SignalRW[str], PvSuffix.rbv("Manufacturer")]
    model: A[SignalRW[str], PvSuffix.rbv("Model")]
    serial_number: A[SignalRW[str], PvSuffix.rbv("SerialNumber")]
    firmware_version: A[SignalRW[str], PvSuffix.rbv("FirmwareVersion")]
    sdk_version: A[SignalRW[str], PvSuffix.rbv("SDKVersion")]
    driver_version: A[SignalRW[str], PvSuffix.rbv("DriverVersion")]

    # Detector Information
    description: A[SignalR[str], PvSuffix("Description_RBV")]
    x_pixel_size: A[SignalR[float], PvSuffix("XPixelSize_RBV")]
    y_pixel_size: A[SignalR[float], PvSuffix("YPixelSize_RBV")]
    sensor_material: A[SignalR[str], PvSuffix("SensorMaterial_RBV")]
    sensor_thickness: A[SignalR[float], PvSuffix("SensorThickness_RBV")]
    dead_time: A[SignalR[float], PvSuffix("DeadTime_RBV")]

    # Detector Status
    state: A[SignalR[str], PvSuffix("State_RBV")]
    error: A[SignalR[str], PvSuffix("Error_RBV")]
    temp0: A[SignalR[float], PvSuffix("Temp0_RBV")]
    humid0: A[SignalR[float], PvSuffix("Humid0_RBV")]
    link0: A[SignalR[str], PvSuffix("Link0_RBV")]
    link1: A[SignalR[str], PvSuffix("Link1_RBV")]
    link2: A[SignalR[str], PvSuffix("Link2_RBV")]
    link3: A[SignalR[str], PvSuffix("Link3_RBV")]
    dcu_buffer_free: A[SignalR[int], PvSuffix("DCUBufferFree_RBV")]
    hv_reset_time: A[SignalRW[float], PvSuffix.rbv("HVResetTime")]
    hv_reset: A[SignalRW[bool], PvSuffix("HVReset", "HVReset")]
    hv_state: A[SignalR[str], PvSuffix("HVState_RBV")]

    # Acquisition Setup
    threshold: A[SignalRW[float], PvSuffix.rbv("Threshold")]
    threshold1_enable: A[SignalRW[bool], PvSuffix.rbv("Threshold1Enable")]
    threshold2: A[SignalRW[float], PvSuffix.rbv("Threshold2")]
    threshold2_enable: A[SignalRW[bool], PvSuffix.rbv("Threshold2Enable")]
    threshold_diff_enable: A[SignalRW[bool], PvSuffix.rbv("ThresholdDiffEnable")]
    photon_energy: A[SignalRW[float], PvSuffix.rbv("PhotonEnergy")]
    counting_mode: A[SignalRW[str], PvSuffix.rbv("CountingMode")]

    # Trigger Setup
    ext_gate_mode: A[SignalRW[str], PvSuffix.rbv("ExtGateMode")]
    trigger: A[SignalRW[float], PvSuffix("Trigger")]
    trigger_exposure: A[SignalRW[float], PvSuffix.rbv("TriggerExposure")]
    num_triggers: A[SignalRW[int], PvSuffix.rbv("NumTriggers")]
    manual_trigger: A[SignalRW[bool], PvSuffix.rbv("ManualTrigger")]
    trigger_start_delay: A[SignalRW[float], PvSuffix.rbv("TriggerStartDelay")]

    # Readout Setup
    roi_mode: A[SignalRW[EigerROIMode], PvSuffix.rbv("ROIMode")]
    flatfield_applied: A[SignalRW[bool], PvSuffix.rbv("FlatfieldApplied")]
    countrate_corr_applied: A[SignalRW[bool], PvSuffix.rbv("CountrateCorrApplied")]
    pixel_mask_applied: A[SignalRW[bool], PvSuffix.rbv("PixelMaskApplied")]
    auto_summation: A[SignalRW[bool], PvSuffix.rbv("AutoSummation")]
    signed_data: A[SignalRW[bool], PvSuffix.rbv("SignedData")]
    compression_algo: A[SignalRW[EigerCompressionAlgo], PvSuffix.rbv("CompressionAlgo")]
    data_source: A[SignalRW[EigerDataSource], PvSuffix.rbv("DataSource")]

    # Acquisition Status
    armed: A[SignalR[bool], PvSuffix("Armed")]
    bit_depth_image: A[SignalR[int], PvSuffix("BitDepthImage_RBV")]
    count_cutoff: A[SignalR[float], PvSuffix("CountCutoff_RBV")]

    # FileWriter Interface
    fw_enable: A[SignalRW[bool], PvSuffix.rbv("FWEnable")]
    fw_state: A[SignalR[str], PvSuffix("FWState_RBV")]
    fw_hdf5_format: A[SignalRW[EigerHDF5Format], PvSuffix.rbv("FWHDF5Format")]
    fw_compression: A[SignalRW[bool], PvSuffix.rbv("FWCompression")]
    fw_nimgs_per_file: A[SignalRW[float], PvSuffix.rbv("FWNImgsPerFile")]
    fw_name_pattern: A[SignalRW[str], PvSuffix.rbv("FWNamePattern")]
    sequence_id: A[SignalR[int], PvSuffix("SequenceId")]
    save_files: A[SignalRW[bool], PvSuffix.rbv("SaveFiles")]
    file_path: A[SignalRW[str], PvSuffix.rbv("FilePath")]
    file_owner: A[SignalRW[str], PvSuffix.rbv("FileOwner")]
    file_owner_grp: A[SignalRW[str], PvSuffix.rbv("FileOwnerGrp")]
    file_perms: A[SignalRW[str], PvSuffix.rbv("FilePerms")]
    fw_free: A[SignalR[float], PvSuffix("FWFree_RBV")]
    fw_auto_remove: A[SignalRW[bool], PvSuffix.rbv("FWAutoRemove")]
    fw_clear: A[SignalRW[float], PvSuffix("FWClear")]

    # Stream Interface
    stream_enable: A[SignalRW[bool], PvSuffix.rbv("StreamEnable")]
    stream_state: A[SignalR[str], PvSuffix("StreamState_RBV")]
    stream_version: A[SignalRW[EigerStreamVersion], PvSuffix.rbv("StreamVersion")]
    stream_decompress: A[SignalRW[bool], PvSuffix.rbv("StreamDecompress")]
    stream_hdr_detail: A[
        SignalRW[EigerStreamHdrDetail], PvSuffix.rbv("StreamHdrDetail")
    ]
    stream_hdr_appendix: A[SignalRW[str], PvSuffix.rbv("StreamHdrAppendix")]
    stream_img_appendix: A[SignalRW[str], PvSuffix.rbv("StreamImgAppendix")]
    stream_dropped: A[SignalR[float], PvSuffix("StreamDropped_RBV")]

    # Monitor Interface
    monitor_enable: A[SignalRW[bool], PvSuffix.rbv("MonitorEnable")]
    monitor_state: A[SignalR[str], PvSuffix("MonitorState_RBV")]
    monitor_timeout: A[SignalRW[float], PvSuffix.rbv("MonitorTimeout")]

    # Acquisition Metadata
    beam_x: A[SignalRW[float], PvSuffix.rbv("BeamX")]
    beam_y: A[SignalRW[float], PvSuffix.rbv("BeamY")]
    det_dist: A[SignalRW[float], PvSuffix.rbv("DetDist")]
    wavelength: A[SignalRW[float], PvSuffix.rbv("Wavelength")]

    # Detector Metadata
    chi_start: A[SignalRW[float], PvSuffix.rbv("ChiStart")]
    chi_incr: A[SignalRW[float], PvSuffix.rbv("ChiIncr")]
    kappa_start: A[SignalRW[float], PvSuffix.rbv("KappaStart")]
    kappa_incr: A[SignalRW[float], PvSuffix.rbv("KappaIncr")]
    omega_start: A[SignalRW[float], PvSuffix.rbv("OmegaStart")]
    omega_incr: A[SignalRW[float], PvSuffix.rbv("OmegaIncr")]
    phi_start: A[SignalRW[float], PvSuffix.rbv("PhiStart")]
    phi_incr: A[SignalRW[float], PvSuffix.rbv("PhiIncr")]
    two_theta_start: A[SignalRW[float], PvSuffix.rbv("TwoThetaStart")]
    two_theta_incr: A[SignalRW[float], PvSuffix.rbv("TwoThetaIncr")]

    # Minimum change allowed
    wavelength_eps: A[SignalRW[float], PvSuffix.rbv("WavelengthEps")]
    energy_eps: A[SignalRW[float], PvSuffix.rbv("EnergyEps")]


class EigerWriter(DetectorWriter):
    """Eiger-specific file writer using the built-in FileWriter interface."""

    def __init__(
        self,
        driver: EigerDriverIO,
        path_provider: PathProvider,
        dataset_describer: DatasetDescriber,
    ):
        self._driver = driver
        self._path_provider = path_provider
        self._dataset_describer = dataset_describer

        self._file_info: PathInfo | None = None
        self._current_sequence_id: int | None = None
        self._composer: HDFDocumentComposer | None = None
        self._exposures_per_event: int = 1

    async def open(self, name: str, exposures_per_event: int = 1) -> dict[str, DataKey]:
        """Setup file writing for acquisition."""
        # Get file path info from path provider
        self._file_info = self._path_provider()
        self._current_sequence_id = await self._driver.sequence_id.get_value()

        # Set the name pattern with $id replacement similar to original
        name_pattern = f"{self._file_info.filename}_$id"

        # Configure the Eiger FileWriter
        await asyncio.gather(
            self._driver.file_path.set(self._file_info.directory_path.as_posix()),
            self._driver.fw_name_pattern.set(name_pattern),
            self._driver.fw_enable.set(True),
            self._driver.save_files.set(True),
            self._driver.fw_hdf5_format.set(EigerHDF5Format.LEGACY),
            self._driver.data_source.set(EigerDataSource.FILE_WRITER),
        )

        # Get important values from the detector
        num_images, num_images_per_file, num_triggers = await asyncio.gather(
            self._driver.num_images.get_value(),
            self._driver.fw_nimgs_per_file.get_value(),
            self._driver.num_triggers.get_value(),
        )

        # Exposures per event is a combination of multiple signals, so we can't simply
        # set it on the detector, unlike other detector implementations
        if num_images != exposures_per_event:
            msg = (
                "Mismatch between the number of images set in the detector "
                "and the expected number of exposures per event. "
                f"Got {num_images} but expected {exposures_per_event}."
            )
            raise ValueError(msg)

        detector_shape, np_dtype = await asyncio.gather(
            self._dataset_describer.shape(),
            self._dataset_describer.np_datatype(),
        )

        # Add the master file datasets
        master_datasets = [
            HDFDatasetDescription(
                data_key=f"{name}_y_pixel_size",
                dataset="entry/instrument/detector/y_pixel_size",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_x_pixel_size",
                dataset="entry/instrument/detector/x_pixel_size",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_detector_distance",
                dataset="entry/instrument/detector/distance",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_incident_wavelength",
                dataset="entry/instrument/detector/incident_wavelength",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_frame_time",
                dataset="entry/instrument/detector/frame_time",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_beam_center_x",
                dataset="entry/instrument/detector/beam_center_x",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_beam_center_y",
                dataset="entry/instrument/detector/beam_center_y",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_count_time",
                dataset="entry/instrument/detector/count_time",
                shape=(exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_pixel_mask",
                dataset="entry/instrument/detector/detectorSpecific/pixel_mask",
                # TODO: Maybe only 1 mask?
                shape=(exposures_per_event, *detector_shape),
                dtype_numpy=np.dtype(np.uint8).str,
                chunk_shape=(1, *detector_shape),
            ),
        ]

        # Cache for use later
        self._exposures_per_event = exposures_per_event

        # Add the array datasets (linked from the master file)
        # Linked keys are of the form
        # - "/entry/data_000001"
        # - "/entry/data_000002"
        # - ...
        # Example: if exposures_per_event (num_images) is 60, num_triggers is 2, and num_images_per_file is 100,
        # then the data_000001 file will have 100 images and the data_000002 filewill have 20 images.
        # Put simply, the last file could have less than num_images_per_file images.
        total_images = num_triggers * exposures_per_event
        frame_datasets = [
            HDFDatasetDescription(
                data_key=f"{name}_{i}",
                dataset=f"/entry/data_{i:06d}",
                shape=(
                    min(
                        num_images_per_file,
                        total_images - (i - 1) * num_images_per_file,
                    ),
                    *detector_shape,
                ),
                dtype_numpy=np_dtype,
                chunk_shape=(1, *detector_shape),
            )
            for i in range(1, ceil(total_images / num_images_per_file) + 1)
        ]

        self._datasets = master_datasets + frame_datasets

        return {
            ds.data_key: DataKey(
                source=self._master_file_path.as_posix(),
                shape=list(ds.shape),
                dtype="array"
                if exposures_per_event > 1 or len(ds.shape) > 1
                else "number",
                dtype_numpy=ds.dtype_numpy,
                external="STREAM:",
            )
            for ds in self._datasets
        }

    @property
    def _master_file_path(self) -> Path:
        if self._current_sequence_id is None or self._file_info is None:
            msg = "Must call EigerWriter.open() before accessing master file path"
            raise ValueError(msg)
        return (
            self._file_info.directory_path
            / f"{self._file_info.filename}_{self._current_sequence_id}_master.h5"
        )

    async def compute_index(self, num_images_counter: int) -> int:
        if await self._driver.num_images.get_value() != self._exposures_per_event:
            msg = "Detected change to the number of images during acquisition. This is not allowed."
            raise RuntimeError(msg)
        return num_images_counter // self._exposures_per_event

    async def observe_indices_written(
        self, timeout: float = 10.0
    ) -> AsyncGenerator[int, None]:
        """Monitor the number of files written by the Eiger FileWriter."""
        async for num_images_counter in observe_value(
            self._driver.num_images_counter, timeout
        ):
            yield await self.compute_index(num_images_counter)

    async def get_indices_written(self) -> int:
        """Get the current number of indices written.

        Since Eiger defines the number of triggers, we want each trigger
        to correspond to a single index. This is in contrast to the ophyd-sync
        implementation, where the `num_triggers` was ignored.
        """
        return await self.compute_index(
            await self._driver.num_images_counter.get_value()
        )

    async def collect_stream_docs(
        self, _name: str, indices_written: int
    ) -> AsyncIterator[StreamAsset]:
        """Generate stream documents for the written HDF5 files."""
        if indices_written:
            if not self._composer:
                self._composer = HDFDocumentComposer(
                    self._master_file_path,
                    self._datasets,
                )

                for doc in self._composer.stream_resources():
                    yield "stream_resource", doc

            for doc in self._composer.stream_data(indices_written):
                yield "stream_datum", doc

    async def close(self) -> None:
        """Clean up file writing after acquisition and validate files exist."""
        # Disable file writer
        await asyncio.gather(
            self._driver.fw_enable.set(False),
            self._driver.save_files.set(False),
        )

        if not self._master_file_path.exists():
            logger.warning("Master file was not written: %s", self._master_file_path)

        self._composer = None
        self._file_info = None
        self._current_sequence_id = None


class EigerController(ADBaseController[EigerDriverIO]):
    """Controller for Eiger detector, handling trigger modes and acquisition setup."""

    def __init__(
        self, driver: EigerDriverIO, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(driver, *args, **kwargs)

    def get_deadtime(self, _exposure: float | None) -> float:
        """Get detector deadtime for the given exposure."""
        return 0.001

    async def prepare(self, trigger_info: TriggerInfo) -> None:
        """Prepare the detector for acquisition."""
        if (exposure := trigger_info.livetime) is not None:
            await self._driver.acquire_time.set(exposure)

        # Configure trigger mode based on TriggerInfo
        if trigger_info.trigger == DetectorTrigger.INTERNAL:
            await self._driver.trigger_mode.set(EigerTriggerMode.INTERNAL_SERIES)
        elif trigger_info.trigger == DetectorTrigger.EXTERNAL:
            await self._driver.trigger_mode.set(EigerTriggerMode.EXTERNAL_SERIES)
        elif trigger_info.trigger in [
            DetectorTrigger.VARIABLE_GATE,
            DetectorTrigger.CONSTANT_GATE,
        ]:
            await self._driver.trigger_mode.set(EigerTriggerMode.EXTERNAL_GATE)
        else:
            msg = f"Trigger mode {trigger_info.trigger} not supported"
            raise NotImplementedError(msg)

        if trigger_info.total_number_of_exposures == 0:
            image_mode = ADImageMode.CONTINUOUS
        else:
            image_mode = ADImageMode.MULTIPLE

        await asyncio.gather(
            self._driver.num_triggers.set(trigger_info.number_of_events),
            self._driver.num_images.set(trigger_info.exposures_per_event),
            self._driver.image_mode.set(image_mode),
        )


class EigerDetector(AreaDetector[EigerController]):
    """Eiger detector implementation using AreaDetector pattern."""

    def __init__(
        self,
        prefix: str,
        path_provider: PathProvider,
        driver_suffix: str = "cam1:",
        writer_cls: type[DetectorWriter] = EigerWriter,
        fileio_suffix: str | None = None,
        name: str = "",
        config_sigs: Sequence[SignalR] = (),
        plugins: dict[str, NDPluginBaseIO] | None = None,
    ):
        driver = EigerDriverIO(prefix + driver_suffix)
        controller = EigerController(driver)

        # If the writer class is an ADWriter, use the with_io method
        # since we want to use one of the AD plugins.
        # Otherwise, use the internal file writer.
        if issubclass(writer_cls, ADWriter):
            writer = writer_cls.with_io(
                prefix,
                path_provider,
                dataset_source=driver,
                fileio_suffix=fileio_suffix,
                plugins=plugins,
            )
        else:
            if fileio_suffix is not None or plugins is not None:
                logger.warning(
                    "Ignoring params fileio_suffix and plugins for non-ADWriter writer"
                )
            writer = writer_cls(driver, path_provider)

        super().__init__(
            controller=controller,
            writer=writer,
            plugins=plugins,
            name=name,
            config_sigs=config_sigs,
        )
