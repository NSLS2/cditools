"""
Tests for the EigerWriter class using ophyd-async mocking utilities.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import os
import shutil

import h5py
import numpy as np
import pytest
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
import bluesky.plan_stubs as bps
from bluesky.utils import MsgGenerator
from ophyd_async.core import (
    DetectorTrigger,
    PathProvider,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    init_devices,
)
from ophyd_async.epics.adcore import ADBaseDatasetDescriber, ADImageMode
from ophyd_async.testing import (
    set_mock_value,
)

from tiled.server.simple import SimpleTiledServer
from tiled.client.container import Container

from cditools.eiger_async import (
    EigerController,
    EigerDataSource,
    EigerDetector,
    EigerDriverIO,
    EigerFileIO,
    EigerHDF5Format,
    EigerTriggerMode,
    EigerWriter,
)


def write_eiger_hdf5_file(num_triggers: int, num_images: int, sequence_id: int, name: str = "test_eiger"):
    if not os.path.exists(f"/tmp/test_data/"):
        os.makedirs(f"/tmp/test_data/")

    with h5py.File(f"/tmp/test_data/{name}_{sequence_id}_data_000001.h5", "w") as f:
        f.create_dataset("data_000001", data=np.zeros((num_triggers * num_images, 2048, 2048), dtype=np.uint16))

    with h5py.File(f"/tmp/test_data/{name}_{sequence_id}_master.h5", "w") as f:
        f["entry/data/data_000001"] = h5py.ExternalLink(f"/tmp/test_data/{name}_{sequence_id}_data_000001.h5", "data_000001")
        f.create_dataset("entry/instrument/detector/y_pixel_size", data=np.ones((num_images,), dtype=np.uint8))
        f.create_dataset("entry/instrument/detector/x_pixel_size", data=np.ones((num_images,), dtype=np.uint8))
        f.create_dataset("entry/instrument/detector/distance", data=np.ones((num_images,), dtype=np.float32))
        f.create_dataset("entry/instrument/detector/incident_wavelength", data=np.ones((num_images,), dtype=np.float32))
        f.create_dataset("entry/instrument/detector/frame_time", data=np.ones((num_images,), dtype=np.float32))
        f.create_dataset("entry/instrument/detector/beam_center_x", data=np.ones((num_images,), dtype=np.uint8))
        f.create_dataset("entry/instrument/detector/beam_center_y", data=np.ones((num_images,), dtype=np.uint8))
        f.create_dataset("entry/instrument/detector/count_time", data=np.ones((num_images,), dtype=np.float32))
        f.create_dataset("entry/instrument/detector/pixel_mask", data=np.zeros((num_images, 2048, 2048), dtype=np.uint8))

@pytest.fixture
def mock_eiger_detector(RE: RunEngine) -> EigerDetector:
    path_provider = StaticPathProvider(
        StaticFilenameProvider("test_eiger"), directory_path="/tmp/test_data"
    )
    with init_devices(mock=True):
        detector = EigerDetector("MOCK:EIGER:", path_provider, name="test_eiger")
    set_mock_value(detector.driver.array_size_x, 2048)
    set_mock_value(detector.driver.array_size_y, 2048)
    set_mock_value(detector.driver.data_type, "UInt16")

    yield detector

    if os.path.exists("/tmp/test_data"):
        shutil.rmtree("/tmp/test_data")


@pytest.fixture
def mock_eiger_driver(RE: RunEngine) -> EigerDriverIO:
    """Create a mock EigerDriverIO for testing."""
    with init_devices(mock=True):
        driver = EigerDriverIO("MOCK:EIGER:cam1:")

    # Set up some default mock values
    set_mock_value(driver.array_size_x, 2048)
    set_mock_value(driver.array_size_y, 2048)
    set_mock_value(driver.data_type, "UInt16")

    return driver


@pytest.fixture
def mock_eiger_fileio(RE: RunEngine) -> EigerFileIO:
    with init_devices(mock=True):
        fileio = EigerFileIO("MOCK:EIGER:cam1:")

    return fileio


@pytest.fixture
def mock_path_provider() -> PathProvider:
    """Create a mock path provider for testing."""
    return StaticPathProvider(
        StaticFilenameProvider("test_eiger"), directory_path="/tmp/test_data"
    )


@pytest.fixture
def eiger_writer(
    mock_eiger_driver: EigerDriverIO,
    mock_eiger_fileio: EigerFileIO,
    mock_path_provider: PathProvider,
) -> EigerWriter:
    """Create an EigerWriter instance for testing."""
    dataset_describer = ADBaseDatasetDescriber(mock_eiger_driver)
    return EigerWriter(mock_eiger_fileio, mock_path_provider, dataset_describer)


@pytest.fixture
def eiger_controller(mock_eiger_driver: EigerDriverIO) -> EigerController:
    return EigerController(mock_eiger_driver)


@pytest.mark.asyncio
async def test_eiger_writer_initialization(
    eiger_writer: EigerWriter,
    mock_eiger_fileio: EigerFileIO,
    mock_path_provider: PathProvider,
):
    """Test that EigerWriter initializes correctly."""
    assert eiger_writer.fileio is mock_eiger_fileio
    assert eiger_writer._path_provider is mock_path_provider
    assert eiger_writer._dataset_describer is not None
    assert eiger_writer._file_info is None
    assert eiger_writer._current_sequence_id is None
    assert eiger_writer._composer is None


@pytest.mark.asyncio
async def test_eiger_writer_open(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
    mock_eiger_fileio: EigerFileIO,
) -> None:
    """Test the open method configures the detector correctly."""
    array_size_x, array_size_y, data_type = await asyncio.gather(
        mock_eiger_driver.array_size_x.get_value(),
        mock_eiger_driver.array_size_y.get_value(),
        mock_eiger_driver.data_type.get_value(),
    )

    # Case 1: 1 image per file, 1 image, 1 trigger
    set_mock_value(mock_eiger_fileio.sequence_id, 1)
    set_mock_value(mock_eiger_fileio.fw_nimgs_per_file, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)

    description = await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    assert await mock_eiger_fileio.fw_enable.get_value() is True
    assert await mock_eiger_fileio.save_files.get_value() is True
    assert await mock_eiger_fileio.fw_hdf5_format.get_value() == EigerHDF5Format.LEGACY
    assert description.keys() == {
        "test_eiger_y_pixel_size",
        "test_eiger_x_pixel_size",
        "test_eiger_detector_distance",
        "test_eiger_incident_wavelength",
        "test_eiger_frame_time",
        "test_eiger_beam_center_x",
        "test_eiger_beam_center_y",
        "test_eiger_count_time",
        "test_eiger_pixel_mask",
        "test_eiger_1",
    }
    assert (
        description["test_eiger_1"]["source"] == "/tmp/test_data/test_eiger_1_master.h5"
    )

    # Case 2: 4 images per file, 11 images, 2 triggers
    # Expect 6 files, the first 5 will have 4 images, the last will have 2
    set_mock_value(mock_eiger_fileio.sequence_id, 2)
    set_mock_value(mock_eiger_fileio.fw_nimgs_per_file, 4)
    set_mock_value(mock_eiger_driver.num_images, 11)
    set_mock_value(mock_eiger_driver.num_triggers, 2)
    description = await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert description.keys() == {
        "test_eiger_y_pixel_size",
        "test_eiger_x_pixel_size",
        "test_eiger_detector_distance",
        "test_eiger_incident_wavelength",
        "test_eiger_frame_time",
        "test_eiger_beam_center_x",
        "test_eiger_beam_center_y",
        "test_eiger_count_time",
        "test_eiger_pixel_mask",
        "test_eiger_1",
    }
    data_key = description["test_eiger_1"]
    assert tuple(data_key["shape"]) == (11, array_size_x, array_size_y)
    assert data_key["dtype"] == "array"
    assert data_key["dtype_numpy"] == np.dtype(data_type.lower()).str
    assert data_key["external"] == "STREAM:"
    assert data_key["source"] == "/tmp/test_data/test_eiger_2_master.h5"

    #     # TODO: Add back when nimages_per_file is no longer hardcoded
    #     "test_eiger_2",
    #     "test_eiger_3",
    #     "test_eiger_4",
    #     "test_eiger_5",
    #     "test_eiger_6",
    # }

    # for i in range(1, 6):
    #     data_key = description[f"test_eiger_{i}"]
    #     assert tuple(data_key["shape"]) == (4, array_size_x, array_size_y)
    #     assert data_key["dtype"] == "array"
    #     assert data_key["dtype_numpy"] == np.dtype(data_type.lower()).str
    #     assert data_key["external"] == "STREAM:"
    #     assert data_key["source"] == "/tmp/test_data/test_eiger_2_master.h5"
    # data_key = description["test_eiger_6"]
    # assert tuple(data_key["shape"]) == (2, array_size_x, array_size_y)
    # assert data_key["dtype"] == "array"
    # assert data_key["dtype_numpy"] == np.dtype(data_type.lower()).str
    # assert data_key["external"] == "STREAM:"
    # assert data_key["source"] == "/tmp/test_data/test_eiger_2_master.h5"


@pytest.mark.asyncio
async def test_eiger_writer_get_indices_written(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
    mock_eiger_fileio: EigerFileIO,
):
    """Test getting the number of indices written."""
    set_mock_value(mock_eiger_fileio.sequence_id, 1)
    set_mock_value(mock_eiger_fileio.fw_nimgs_per_file, 1)

    # Case 1: 1 image, 1 trigger
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_fileio.num_captured, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_fileio.num_captured, 1)
    assert await eiger_writer.get_indices_written() == 1

    # Case 2: 1 image, 5 triggers
    set_mock_value(mock_eiger_driver.num_triggers, 5)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_fileio.num_captured, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_fileio.num_captured, 1)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_fileio.num_captured, 3)
    assert await eiger_writer.get_indices_written() == 3
    set_mock_value(mock_eiger_fileio.num_captured, 5)
    assert await eiger_writer.get_indices_written() == 5

    # Case 3: 5 images, 2 triggers
    set_mock_value(mock_eiger_driver.num_triggers, 2)
    set_mock_value(mock_eiger_driver.num_images, 5)
    set_mock_value(mock_eiger_fileio.num_captured, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_fileio.num_captured, 4)
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_fileio.num_captured, 5)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_fileio.num_captured, 9)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_fileio.num_captured, 10)
    assert await eiger_writer.get_indices_written() == 2


@pytest.mark.asyncio
async def test_eiger_writer_observe_indices_written(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
    mock_eiger_fileio: EigerFileIO,
) -> None:
    """Test observing indices as they are written."""
    set_mock_value(mock_eiger_fileio.sequence_id, 1)
    set_mock_value(mock_eiger_fileio.fw_nimgs_per_file, 1)

    async def _simulate_writing_indices(
        num_images: int, num_triggers: int, acquire_time: float = 0.01
    ) -> list[int]:
        # Create an async generator to track yielded indices
        observed_indices = []
        set_mock_value(mock_eiger_fileio.num_captured, 0)

        async def _simulate_acquisition():
            """Simulate the detector writing images by incrementing sequence_id."""
            for i in range(1, num_images * num_triggers + 1):
                await asyncio.sleep(acquire_time)
                set_mock_value(mock_eiger_fileio.num_captured, i)

        async def _complete():
            """Helper function to collect observed indices."""
            indices_written = eiger_writer.observe_indices_written(timeout=1.0)
            async for index in indices_written:
                observed_indices.append(index)
                if index >= num_triggers:
                    break

        # Start the simulation task
        sim_task = asyncio.create_task(_simulate_acquisition())

        # Observe the indices being written
        observe_task = asyncio.create_task(_complete())

        # Wait for both tasks to complete
        await asyncio.gather(sim_task, observe_task)

        return observed_indices

    # Case 1: 1 image, 1 trigger
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    num_images = await mock_eiger_driver.num_images.get_value()
    num_triggers = await mock_eiger_driver.num_triggers.get_value()
    await eiger_writer.open(name="test_eiger", exposures_per_event=num_images)
    observed = await _simulate_writing_indices(
        num_images=num_images, num_triggers=num_triggers
    )
    assert observed == [0, 1]

    # Case 2: 1 image, 5 triggers
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 5)
    num_images = await mock_eiger_driver.num_images.get_value()
    num_triggers = await mock_eiger_driver.num_triggers.get_value()
    await eiger_writer.open(name="test_eiger", exposures_per_event=num_images)
    observed = await _simulate_writing_indices(
        num_images=num_images, num_triggers=num_triggers
    )
    assert observed == [0, 1, 2, 3, 4, 5]

    # Case 3: 5 images, 2 triggers
    set_mock_value(mock_eiger_driver.num_images, 5)
    set_mock_value(mock_eiger_driver.num_triggers, 2)
    num_images = await mock_eiger_driver.num_images.get_value()
    num_triggers = await mock_eiger_driver.num_triggers.get_value()
    await eiger_writer.open(name="test_eiger", exposures_per_event=num_images)
    observed = await _simulate_writing_indices(
        num_images=num_images, num_triggers=num_triggers
    )
    assert observed == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]


@pytest.mark.asyncio
async def test_eiger_writer_collect_stream_docs(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
    mock_eiger_fileio: EigerFileIO,
) -> None:
    """Test collecting stream documents."""

    async def collect_docs(num_triggers: int):
        resource_docs = []
        data_docs = []
        for i in range(1, num_triggers + 1):
            async for doc_type, doc in eiger_writer.collect_stream_docs(
                _name="", indices_written=i
            ):
                if doc_type == "stream_resource":
                    resource_docs.append(doc)
                elif doc_type == "stream_datum":
                    data_docs.append(doc)
        return resource_docs, data_docs

    set_mock_value(mock_eiger_fileio.sequence_id, 1)
    set_mock_value(mock_eiger_fileio.fw_nimgs_per_file, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    resource_docs, data_docs = await collect_docs(
        num_triggers=await mock_eiger_driver.num_triggers.get_value()
    )
    assert len(resource_docs) == 10
    assert len(data_docs) == 10
    assert (
        resource_docs[0]["uri"]
        == "file://localhost/tmp/test_data/test_eiger_1_master.h5"
    )

    await eiger_writer.close()

    set_mock_value(mock_eiger_driver.num_triggers, 3)
    set_mock_value(mock_eiger_fileio.sequence_id, 2)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    resource_docs, data_docs = await collect_docs(
        num_triggers=await mock_eiger_driver.num_triggers.get_value()
    )
    assert len(resource_docs) == 10
    assert len(data_docs) == 30
    assert (
        resource_docs[0]["uri"]
        == "file://localhost/tmp/test_data/test_eiger_2_master.h5"
    )


@pytest.mark.asyncio
async def test_eiger_writer_close(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
    mock_eiger_fileio: EigerFileIO,
) -> None:
    """Test closing the writer."""

    # Verify the writing was enabled
    set_mock_value(mock_eiger_fileio.sequence_id, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    set_mock_value(mock_eiger_fileio.fw_nimgs_per_file, 1)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    assert await mock_eiger_fileio.fw_enable.get_value() is True
    assert await mock_eiger_fileio.save_files.get_value() is True

    # Verify the writing was disabled
    await eiger_writer.close()
    assert await mock_eiger_fileio.fw_enable.get_value() is False
    assert await mock_eiger_fileio.save_files.get_value() is False
    assert eiger_writer._composer is None
    assert eiger_writer._current_sequence_id is None
    assert eiger_writer._file_info is None
    assert await mock_eiger_fileio.fw_nimgs_per_file.get_value() == 1


# TODO: Test the controller's overridden methods and do an integration test with bluesky plans + tiled readback
@pytest.mark.asyncio
async def test_eiger_controller_prepare(eiger_controller: EigerController) -> None:
    trigger_info = TriggerInfo(
        number_of_events=1,
        livetime=0.01,
        deadtime=0.001,
        trigger=DetectorTrigger.INTERNAL,
        exposure_timeout=1.0,
        exposures_per_event=1,
    )
    await eiger_controller.prepare(trigger_info)
    assert await eiger_controller.driver.acquire_time.get_value() == 0.01
    assert (
        await eiger_controller.driver.trigger_mode.get_value()
        == EigerTriggerMode.INTERNAL_SERIES
    )
    assert await eiger_controller.driver.num_triggers.get_value() == 1
    assert await eiger_controller.driver.num_images.get_value() == 1
    assert await eiger_controller.driver.image_mode.get_value() == ADImageMode.MULTIPLE

    trigger_info = TriggerInfo(
        number_of_events=10,
        livetime=0.0,
        deadtime=0.0,
        trigger=DetectorTrigger.EDGE_TRIGGER,
        exposure_timeout=10.0,
        exposures_per_event=5,
    )
    await eiger_controller.prepare(trigger_info)
    assert await eiger_controller.driver.acquire_time.get_value() == 0.0
    assert (
        await eiger_controller.driver.trigger_mode.get_value()
        == EigerTriggerMode.EXTERNAL_SERIES
    )
    assert await eiger_controller.driver.num_triggers.get_value() == 10
    assert await eiger_controller.driver.num_images.get_value() == 5
    assert await eiger_controller.driver.image_mode.get_value() == ADImageMode.MULTIPLE

    trigger_info = TriggerInfo(
        number_of_events=0,
        livetime=None,
        deadtime=0.0,
        trigger=DetectorTrigger.VARIABLE_GATE,
        exposure_timeout=10.0,
        exposures_per_event=1,
    )
    await eiger_controller.prepare(trigger_info)
    assert await eiger_controller.driver.acquire_time.get_value() == 0.0
    assert (
        await eiger_controller.driver.trigger_mode.get_value()
        == EigerTriggerMode.EXTERNAL_GATE
    )
    assert await eiger_controller.driver.num_triggers.get_value() == 0
    assert await eiger_controller.driver.num_images.get_value() == 1
    assert (
        await eiger_controller.driver.image_mode.get_value() == ADImageMode.CONTINUOUS
    )


@pytest.mark.asyncio
async def test_eiger_detector(mock_eiger_detector: EigerDetector) -> None:
    set_mock_value(mock_eiger_detector.driver.num_images, 1)
    set_mock_value(mock_eiger_detector.driver.num_triggers, 2)
    set_mock_value(mock_eiger_detector.driver.acquire_period, 0.01)

    async def _simulate_one_trigger(num_captured: int):
        await asyncio.sleep(await mock_eiger_detector.driver.acquire_period.get_value())
        set_mock_value(mock_eiger_detector.fileio.num_captured, num_captured)

    # Standalone methods
    await mock_eiger_detector.describe()

    # Case 1 - Step Scan: stage, trigger, read, trigger, read, unstage
    await mock_eiger_detector.stage()
    assert (
        await mock_eiger_detector.driver.data_source.get_value()
        == EigerDataSource.FILE_WRITER
    )
    status = mock_eiger_detector.trigger()
    await _simulate_one_trigger(1)
    await status
    await mock_eiger_detector.read()
    status = mock_eiger_detector.trigger()
    await _simulate_one_trigger(2)
    await status
    await mock_eiger_detector.read()
    await mock_eiger_detector.unstage()

    # Case 2 - Fly Scan: prepare, kickoff, complete
    await mock_eiger_detector.prepare(
        TriggerInfo(
            number_of_events=1,
            livetime=0.01,
            deadtime=0.001,
            trigger=DetectorTrigger.INTERNAL,
            exposure_timeout=1.0,
            exposures_per_event=1,
        )
    )
    await mock_eiger_detector.kickoff()
    await mock_eiger_detector.complete()


@pytest.mark.asyncio
async def test_eiger_detector_with_RE(RE: RunEngine, tiled_client: Container, mock_eiger_detector: EigerDetector) -> None:
    set_mock_value(mock_eiger_detector.fileio.sequence_id, 1)
    set_mock_value(mock_eiger_detector.driver.num_images, 1)
    set_mock_value(mock_eiger_detector.driver.num_triggers, 1)
    acquire_period = 0.01

    def _count_plan(dets: Sequence[EigerDetector], num: int = 1, num_images: int = 1, sequence_id: int = 1) -> MsgGenerator[str]:
        yield from bps.stage_all(*dets)
        yield from bps.open_run()

        for _ in range(num):
            read_values = {}
            for det in dets:
                read_values[det] = yield from bps.rd(det.fileio.num_captured)
            
            for det in dets:
                yield from bps.trigger(det, wait=False, group="wait_for_trigger")

            yield from bps.sleep(acquire_period)

            write_eiger_hdf5_file(num_triggers=1, num_images=num_images, sequence_id=sequence_id, name="test_eiger")

            for det in dets:
                num_images = yield from bps.rd(det.driver.num_images)
                set_mock_value(det.fileio.num_captured, read_values[det] + num_images)

            yield from bps.wait(group="wait_for_trigger")
            yield from bps.create()

            for det in dets:
                yield from bps.read(det)

            yield from bps.save()

        yield from bps.close_run()
        yield from bps.unstage_all(*dets)

    tiled_writer = TiledWriter(tiled_client)
    RE.subscribe(tiled_writer)

    uid = RE(_count_plan([mock_eiger_detector]))
    assert uid is not None
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_1"].shape == (1, 2048, 2048)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_1"].dtype == np.uint16
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_x_pixel_size"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_x_pixel_size"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_y_pixel_size"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_y_pixel_size"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_detector_distance"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_detector_distance"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_incident_wavelength"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_incident_wavelength"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_frame_time"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_frame_time"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_x"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_x"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_y"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_y"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_count_time"].shape == (1,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_count_time"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_pixel_mask"].shape == (1, 2048, 2048)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_pixel_mask"].dtype == np.uint8

    set_mock_value(mock_eiger_detector.fileio.sequence_id, 2)
    set_mock_value(mock_eiger_detector.driver.num_images, 5)


    uid = RE(_count_plan([mock_eiger_detector], num=10, num_images=5, sequence_id=2))
    assert uid is not None
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_1"].shape == (50, 2048, 2048)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_1"].dtype == np.uint16
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_x_pixel_size"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_x_pixel_size"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_y_pixel_size"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_y_pixel_size"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_detector_distance"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_detector_distance"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_incident_wavelength"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_incident_wavelength"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_frame_time"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_frame_time"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_x"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_x"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_y"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_beam_center_y"].dtype == np.uint8
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_count_time"].shape == (50,)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_count_time"].dtype == np.float32
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_pixel_mask"].shape == (50, 2048, 2048)
    assert tiled_client.values().last()["streams"]["primary"]["test_eiger_pixel_mask"].dtype == np.uint8