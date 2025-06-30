"""
Ophyd Async implementation for Eiger detector.
"""
from dataclasses import dataclass
import asyncio
import time
from typing import Sequence, Annotated as A, Any
from pathlib import Path
from logging import getLogger

import numpy as np
from event_model import DataKey
from ophyd_async.core import (
    PathProvider, TriggerInfo, SignalRW, SignalR, DetectorTrigger, 
    StandardDetector, Device, DatasetDescriber, DetectorController, DetectorWriter,
    HDFDatasetDescription, HDFDocumentComposer,
)
from ophyd_async.epics.signal import PvSuffix
from ophyd_async.epics.adcore import ADWriter, NDPluginBaseIO, ADBaseIO, AreaDetector, ADBaseController, ADBaseDatasetDescriber, ADImageMode, ADHDFWriter
from ophyd_async.core import StrictEnum

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
    num_triggers: A[SignalRW[float], PvSuffix.rbv("NumTriggers")]
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
    sequence_id: A[SignalR[float], PvSuffix("SequenceId")]
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
    stream_hdr_detail: A[SignalRW[EigerStreamHdrDetail], PvSuffix.rbv("StreamHdrDetail")]
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


@dataclass
class EigerDatasetDescription:
    """A subset of the HDFDatasetDescription dataclass for Eiger.
    
    This is used to describe the different datasets in the master file.

    Attributes:
        dataset (str): The dataset path in the HDF5 file.
        shape (tuple[int, ...]): The shape of the dataset (excluding the first dimension, which is the number of exposures).
        dtype_numpy (str): The numpy dtype of the dataset.
    """
    dataset: str
    shape: tuple[int, ...]
    dtype_numpy: str


class EigerWriter(DetectorWriter):
    """Eiger-specific file writer using the built-in FileWriter interface.
    
    Adapted from the original eiger.py EigerFileHandler implementation
    to work with the ophyd-async DetectorWriter interface.

    TODO: How this should work is that the it should only use a single sequence ID for the whole scan.
    TODO: Use the num_triggers as the total number of events.
    Indices written should be `(num_images // num_triggers) % num_images_counter`.
    """

    def __init__(self, driver: EigerDriverIO, path_provider: PathProvider, dataset_describer: DatasetDescriber):
        self._driver = driver
        self._path_provider = path_provider
        self._dataset_describer = dataset_describer
        self._sequence_id_offset = 1
        self._initial_sequence_id = 1

    async def open(self, name: str, exposures_per_event: int = 1) -> dict[str, Any]:
        """Setup file writing for acquisition."""
        # Get file path info from path provider
        self._file_info = self._path_provider()

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

        # Get the initial sequence ID to use to determine the indices written
        self._initial_sequence_id = await self._driver.sequence_id.get_value()

        # Exposures per event is a combination of multiple signals, so we can't simply
        # set it on the detector, unlike other detector implementations
        self._exposures_per_event = num_images * num_triggers
        if self._exposures_per_event != exposures_per_event:
            msg = ("Mismatch between the calculated exposures per event and the expected exposures per event. "
                   f"Got {self._exposures_per_event} but expected {exposures_per_event}.")
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
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_x_pixel_size",
                dataset="entry/instrument/detector/x_pixel_size",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_detector_distance",
                dataset="entry/instrument/detector/distance",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_incident_wavelength",
                dataset="entry/instrument/detector/incident_wavelength",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_frame_time",
                dataset="entry/instrument/detector/frame_time",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_beam_center_x",
                dataset="entry/instrument/detector/beam_center_x",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_beam_center_y",
                dataset="entry/instrument/detector/beam_center_y",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_count_time",
                dataset="entry/instrument/detector/count_time",
                shape=(self._exposures_per_event,),
                dtype_numpy=np.dtype(np.float32).str,
                chunk_shape=(1,),
            ),
            HDFDatasetDescription(
                data_key=f"{name}_pixel_mask",
                dataset="entry/instrument/detector/detectorSpecific/pixel_mask",
                # TODO: Maybe only 1 mask?
                shape=(self._exposures_per_event, *detector_shape),
                dtype_numpy=np.dtype(np.uint8).str,
                chunk_shape=(1, *detector_shape),
            ),
        ]

        # Add the array datasets (linked from the master file)
        # Linked keys are of the form
        # - "/entry/data_000001"
        # - "/entry/data_000002"
        # - ...
        # Example: if exposures_per_event (num_images) is 60, num_triggers is 2, and num_images_per_file is 100,
        # then the data_000001 file will have 100 images and the data_000002 filewill have 20 images.
        # Put simply, the last file could have less than num_images_per_file images.
        frame_datasets = [
            HDFDatasetDescription(
                data_key=f"{name}_{i}",
                dataset=f"/entry/data_{i:06d}",
                shape=(min(num_images_per_file, self._exposures_per_event - (i - 1) * num_images_per_file), *detector_shape),
                dtype_numpy=np_dtype,
                chunk_shape=(1, *detector_shape),
            )
            for i in range(1, num_triggers + 1)
        ]

        self._datasets = master_datasets + frame_datasets

        describe = {
            ds.data_key: DataKey(
                source=await self._driver.file_path.get_value(),
                shape=list(ds.shape),
                dtype="array" if self._exposures_per_event > 1 or len(ds.shape) > 1 else "number",
                dtype_numpy=ds.dtype_numpy,
                external="STREAM:",
            )
            for ds in self._datasets
        }

        return describe

    async def observe_indices_written(self, timeout: float = 10.0):
        """Monitor the number of files written by the Eiger FileWriter."""
        last_sequence_id = self._initial_sequence_id
        
        while True:
            try:
                # Check sequence ID to see how many acquisitions have been written
                current_sequence_id = await self._driver.sequence_id.get_value()
                
                if current_sequence_id > last_sequence_id:
                    # Yield indices for all new sequences since last check
                    for seq_id in range(int(last_sequence_id + 1), int(current_sequence_id + 1)):
                        # Convert sequence ID to 0-based index
                        index = seq_id - self._initial_sequence_id - 1
                        yield index
                    last_sequence_id = current_sequence_id
                    
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                break

    async def get_indices_written(self) -> int:
        """Get the current number of indices written."""
        current_sequence_id = await self._driver.sequence_id.get_value()
        return int(current_sequence_id - self._initial_sequence_id)

    async def collect_stream_docs(self, name: str, indices_written: int) -> AsyncIterator[StreamAsset]:
        """Generate stream documents for the written HDF5 files.
        
        Follows the pattern from the original EigerFileHandler.generate_datum method.
        """
        # TODO: Is this needed?
        await self._driver.fw_state.wait_for_value(EigerFileWriterState.IDLE)
        if indices_written:
            if not self._composer:
                # TODO: Use master file path...
                path = Path(await self._driver.file_path.get_value())
                self._composer = HDFDocumentComposer(
                    path,
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
        
        # Validate that master files were written (similar to original unstage method)
        if self._file_prefix is not None and self._initial_sequence_id is not None:
            final_sequence_id = await self._driver.sequence_id.get_value()
            indices_written = int(final_sequence_id - self._initial_sequence_id)
            
            # Check that master files exist for each sequence
            missing_files = []
            for i in range(indices_written):
                sequence_number = self._sequence_id_offset + self._initial_sequence_id + i
                master_file = Path(f"{self._file_prefix}_{int(sequence_number)}_master.h5")
                if not master_file.exists():
                    missing_files.append(master_file)
            
            if missing_files:
                logger.warning(f"Master files were not written: {missing_files}")


class EigerController(ADBaseController[EigerDriverIO]):
    """Controller for Eiger detector, handling trigger modes and acquisition setup."""
    
    def __init__(self, driver: EigerDriverIO, *args: Any, **kwargs: dict[str, Any]) -> None:
        super().__init__(driver, *args, **kwargs)

    def get_deadtime(self, exposure: float | None) -> float:
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
        elif trigger_info.trigger in [DetectorTrigger.VARIABLE_GATE, DetectorTrigger.CONSTANT_GATE]:
            await self._driver.trigger_mode.set(EigerTriggerMode.EXTERNAL_GATE)
        else:
            raise NotImplementedError(f"Trigger mode {trigger_info.trigger} not supported")

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
                logger.warning("Ignoring params fileio_suffix and plugins for non-ADWriter writer")
            writer = writer_cls(driver, path_provider)
        
        super().__init__(
            controller=controller,
            writer=writer,
            plugins=plugins,
            name=name,
            config_sigs=config_sigs,
        )
        