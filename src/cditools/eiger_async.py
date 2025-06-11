"""
Ophyd Async implementation for Eiger detector.
"""
import asyncio
import uuid
import time
from typing import Sequence, Annotated as A
from pathlib import Path

from ophyd_async.core import (
    PathProvider, TriggerInfo, SignalRW, SignalR, DetectorTrigger, 
    StandardDetector, Device, DatasetDescriber, DetectorController
)
from ophyd_async.epics.signal import PvSuffix
from ophyd_async.epics.adcore import ADWriter, NDPluginBaseIO, ADBaseIO, AreaDetector, ADBaseController, ADBaseDatasetDescriber
from ophyd_async.core import StrictEnum


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


class EigerTriggerInfo(TriggerInfo):
    photon_energy: float


class EigerWriter(ADWriter):
    """Eiger-specific file writer using the built-in FileWriter interface."""
    
    def __init__(self, dataset_source: EigerDriverIO, path_provider: PathProvider):
        dataset_describer = ADBaseDatasetDescriber(dataset_source)
        
        # Initialize with empty fileio since Eiger handles its own file writing
        super().__init__(
            fileio=Device(""),  # TODO: Dummy device since Eiger handles file writing internally
            path_provider=path_provider,
            dataset_describer=dataset_describer,
        )
        self._dataset_source = dataset_source
        self._file_info = None
        self._num_captured = 0

    async def open(self, name: str, exposures_per_event: int = 1) -> dict[str, any]:
        """Setup file writing for acquisition."""
        # Get file path info from path provider
        self._file_info = self._path_provider()
        file_path = Path(self._file_info.directory) / self._file_info.filename
        
        # Configure the Eiger FileWriter
        await self._dataset_source.fw_enable.set(True)
        await self._dataset_source.save_files.set(True)
        await self._dataset_source.fw_name_pattern.set(str(file_path.with_suffix(".h5")))
        
        # Set HDF5 format and compression
        await self._dataset_source.fw_hdf5_format.set(EigerHDF5Format.V2024_2)
        await self._dataset_source.fw_compression.set(True)
        
        # Enable data source for file writer
        await self._dataset_source.data_source.set(EigerDataSource.FILE_WRITER)
        
        # Get dataset info for the describe output
        shape = await self._dataset_describer.shape()
        dtype = await self._dataset_describer.np_datatype()
        
        return {
            name: {
                "source": str(file_path.with_suffix(".h5")),
                "shape": shape,
                "dtype": dtype,
                "external": "FILESTORE:" + str(file_path.with_suffix(".h5")),
            }
        }

    async def observe_indices_written(self, timeout: float = 10.0):
        """Monitor the number of files written by the Eiger FileWriter."""
        last_count = 0
        while True:
            try:
                # Check sequence ID to see how many images have been written
                current_count = await self._dataset_source.sequence_id.get_value()
                if current_count > last_count:
                    for i in range(last_count, int(current_count)):
                        yield i
                    last_count = int(current_count)
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                break

    async def get_indices_written(self) -> int:
        """Get the current number of indices written."""
        sequence_id = await self._dataset_source.sequence_id.get_value()
        return int(sequence_id)

    async def collect_stream_docs(self, name: str, indices_written: int):
        """Generate stream documents for the written HDF5 files."""
        if self._file_info is None:
            return
            
        # For Eiger, we typically get one HDF5 file per acquisition
        file_path = Path(self._file_info.directory) / f"{self._file_info.filename}.h5"
        
        # Create stream resource document
        stream_resource = {
            "spec": "EIGER_HDF5",
            "root": str(self._file_info.directory),
            "resource_path": str(file_path.name),
            "resource_kwargs": {},
            "path_semantics": "posix",
            "uid": str(uuid.uuid4()),
        }
        
        yield "stream_resource", stream_resource
        
        # Create stream datum documents for each frame
        for i in range(indices_written):
            stream_datum = {
                "descriptor": name,
                "stream_resource": stream_resource["uid"],
                "seq_num": i,
                "timestamps": [time.time()],
            }
            yield "stream_datum", stream_datum

    async def close(self) -> None:
        """Clean up file writing after acquisition."""
        # Disable file writer
        await self._dataset_source.fw_enable.set(False)
        await self._dataset_source.save_files.set(False)


class EigerController(ADBaseController[EigerDriverIO]):
    """Controller for Eiger detector, handling trigger modes and acquisition setup."""
    
    def __init__(self, driver: EigerDriverIO):
        super().__init__(driver)

    def get_deadtime(self, exposure: float | None) -> float:
        """Get detector deadtime for the given exposure.
        
        For Eiger, deadtime is typically constant and available from the driver.
        """
        # Use the deadtime from the detector, or a default if exposure is None
        return 0.001  # 1ms typical deadtime for Eiger, can be read from driver.dead_time

    async def prepare(self, trigger_info: EigerTriggerInfo) -> None:
        """Prepare the detector for acquisition."""
        # Set photon energy
        await self._driver.photon_energy.set(trigger_info.photon_energy)
        
        # Configure trigger mode based on TriggerInfo
        if trigger_info.trigger == DetectorTrigger.INTERNAL:
            await self._driver.trigger_mode.set(EigerTriggerMode.INTERNAL_SERIES)
        elif trigger_info.trigger == DetectorTrigger.EXTERNAL:
            await self._driver.trigger_mode.set(EigerTriggerMode.EXTERNAL_SERIES)
        else:
            raise NotImplementedError(f"Trigger mode {trigger_info.trigger} not supported")
        
        # Call parent prepare method to handle standard areaDetector setup
        await super().prepare(trigger_info)

    async def arm(self) -> None:
        """Arm the detector for acquisition."""
        # Use parent implementation which handles standard areaDetector arming
        await super().arm()

    async def wait_for_idle(self) -> None:
        """Wait for detector to become idle."""
        # Use parent implementation
        await super().wait_for_idle()

    async def disarm(self) -> None:
        """Disarm the detector after acquisition."""
        # Use parent implementation
        await super().disarm()


class EigerDetector(AreaDetector[EigerController]):
    """Eiger detector implementation using AreaDetector pattern."""

    def __init__(
        self,
        prefix: str,
        path_provider: PathProvider,
        driver_suffix: str = "cam1:",
        name: str = "",
        config_sigs: Sequence[SignalR] = (),
        plugins: dict[str, NDPluginBaseIO] | None = None,
    ):
        # Create driver IO
        driver = EigerDriverIO(prefix + driver_suffix, name="driver")
        
        # Create controller
        controller = EigerController(driver)
        
        # Create writer
        writer = EigerWriter(dataset_source=driver, path_provider=path_provider)
        
        super().__init__(
            controller=controller,
            writer=writer,
            plugins=plugins or {},
            config_sigs=config_sigs,
            name=name,
        )
        
        # Store driver for external access
        self.driver = driver

    async def prepare(self, value: EigerTriggerInfo) -> None:
        """Prepare detector with Eiger-specific trigger info."""
        await super().prepare(value)
