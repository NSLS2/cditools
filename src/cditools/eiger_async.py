"""
Ophyd Async implementation for Eiger detector.
"""
import asyncio
import uuid
import time
from typing import Sequence, Annotated as A, Any
from pathlib import Path
from logging import getLogger
from datetime import datetime

from ophyd_async.core import (
    PathProvider, TriggerInfo, SignalRW, SignalR, DetectorTrigger, 
    StandardDetector, Device, DatasetDescriber, DetectorController, DetectorWriter
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


class EigerWriter(DetectorWriter):
    """Eiger-specific file writer using the built-in FileWriter interface.
    
    Adapted from the original eiger.py EigerFileHandler implementation
    to work with the ophyd-async DetectorWriter interface.
    """
    
    def __init__(self, driver: EigerDriverIO, path_provider: PathProvider):
        self._path_provider = path_provider
        self._driver = driver
        self._sequence_id_offset = 1
        self._resource_uid = None
        self._write_path = None
        self._file_prefix = None
        self._images_per_file = None
        self._resource_kwargs = None
        self._stream_resource_doc = None
        self._initial_sequence_id = None

    async def open(self, multiplier: int = 1) -> dict[str, Any]:
        """Setup file writing for acquisition."""
        # Get file path info from path provider
        self._file_info = self._path_provider()
        
        # Generate resource UID similar to original implementation
        self._resource_uid = str(uuid.uuid4())[:8]  # Similar to new_short_uid()
        
        # Create write path using current datetime
        current_time = datetime.now()
        self._write_path = Path(self._file_info.directory_path) / current_time.strftime("%Y/%m/%d")
        
        # Set up file path on the detector
        await self._driver.file_path.set(str(self._write_path))
        
        # Set the name pattern with $id replacement similar to original
        name_pattern = f"{self._resource_uid}_$id"
        await self._driver.fw_name_pattern.set(name_pattern)
        
        # Configure the Eiger FileWriter
        await asyncio.gather(
            self._driver.fw_enable.set(True),
            self._driver.save_files.set(True),
            self._driver.fw_hdf5_format.set(EigerHDF5Format.V2024_2),
            self._driver.fw_compression.set(True),
            self._driver.data_source.set(EigerDataSource.FILE_WRITER),
        )
        
        # Get images per file for resource document
        self._images_per_file = await self._driver.fw_nimgs_per_file.get_value()
        
        # Set the filename prefix for the resource document
        self._file_prefix = self._write_path / self._resource_uid
        
        # Prepare resource kwargs similar to original implementation
        self._resource_kwargs = {"images_per_file": int(self._images_per_file)}
        
        # Create the stream resource document
        self._stream_resource_doc = {
            "spec": "AD_EIGER",
            "root": str(self._file_info.directory_path),
            "resource_path": str(self._file_prefix.relative_to(self._file_info.directory_path)),
            "resource_kwargs": self._resource_kwargs,
            "path_semantics": "posix",
            "uid": str(uuid.uuid4()),
        }
        
        # Validate and create write path
        if not self._write_path.exists():
            self._write_path.mkdir(parents=True, exist_ok=True)
        
        # Get initial sequence ID to track changes
        self._initial_sequence_id = await self._driver.sequence_id.get_value()
        
        # Get dataset info for the describe output
        array_size_x = await self._driver.array_size_x.get_value()
        array_size_y = await self._driver.array_size_y.get_value()
        data_type = await self._driver.data_type.get_value()
        
        # Convert data type to numpy dtype
        dtype_map = {
            "UInt8": "uint8",
            "UInt16": "uint16", 
            "UInt32": "uint32",
            "Int8": "int8",
            "Int16": "int16",
            "Int32": "int32",
            "Float32": "float32",
            "Float64": "float64",
        }
        dtype = dtype_map.get(data_type, "uint16")  # Default to uint16
        
        return {
            "primary": {
                "source": f"EIGER:{self._resource_uid}",
                "shape": [int(array_size_y), int(array_size_x)],
                "dtype": dtype,
                "external": f"FILESTORE:{self._stream_resource_doc['uid']}",
            }
        }

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

    async def collect_stream_docs(self, indices_written: int):
        """Generate stream documents for the written HDF5 files.
        
        Follows the pattern from the original EigerFileHandler.generate_datum method.
        """
        if self._stream_resource_doc is None:
            return
            
        # Yield the stream resource document first
        yield "stream_resource", self._stream_resource_doc
        
        # Generate stream datum documents for each index written
        for i in range(indices_written):
            # Calculate the actual sequence number like in original implementation
            sequence_number = self._sequence_id_offset + self._initial_sequence_id + i
            
            stream_datum = {
                "descriptor": "primary",
                "stream_resource": self._stream_resource_doc["uid"],
                "seq_num": i,
                "timestamps": [time.time()],
                "datum_kwargs": {"seq_id": int(sequence_number)},
            }
            yield "stream_datum", stream_datum

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
            self._driver.num_images.set(trigger_info.total_number_of_exposures),
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
        