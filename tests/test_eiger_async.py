"""
Tests for the EigerWriter class using ophyd-async mocking utilities.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Generator
from pathlib import Path

import bluesky.plans as bp
import h5py
import numpy as np
import pytest
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from event_model import StreamDatum, StreamResource
from ophyd_async.core import (
    DetectorTrigger,
    PathProvider,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    init_devices,
)
from ophyd_async.epics.adcore import ADBaseDatasetDescriber, ADBaseDataType, ADImageMode
from ophyd_async.testing import (
    callback_on_mock_put,
    set_mock_value,
)
from tiled.client.container import Container

from cditools.eiger_async import (
    EigerController,
    EigerDataSource,
    EigerDetector,
    EigerDriverIO,
    EigerTriggerMode,
    EigerWriter,
)

EIGER_DATA_PATH = Path("/tmp/pytest/eiger_data/")


def write_eiger_hdf5_file(num_images: int, sequence_id: int, name: str = "test_eiger"):
    with h5py.File(f"{EIGER_DATA_PATH}/{name}_{sequence_id}_data_000001.h5", "w") as f:
        f.create_dataset(
            "entry/data/data",
            data=np.zeros((num_images, 2048, 2048), dtype=np.uint32),
        )

    with h5py.File(f"{EIGER_DATA_PATH}/{name}_{sequence_id}_master.h5", "w") as f:
        f["entry/data/data_000001"] = h5py.ExternalLink(
            f"{EIGER_DATA_PATH}/{name}_{sequence_id}_data_000001.h5", "entry/data/data"
        )
        f.create_dataset(
            "entry/instrument/detector/y_pixel_size",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/x_pixel_size",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/detector_distance",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/incident_wavelength",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/frame_time",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/beam_center_x",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/beam_center_y",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/count_time",
            data=np.ones((), dtype=np.float32),
        )
        f.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask",
            data=np.zeros((2048, 2048), dtype=np.uint32),
        )


@pytest.fixture
def mock_eiger_detector(RE: RunEngine) -> Generator[EigerDetector, None, None]:
    if not EIGER_DATA_PATH.exists():
        EIGER_DATA_PATH.mkdir(parents=True)
    path_provider = StaticPathProvider(
        StaticFilenameProvider("test_eiger"), directory_path=EIGER_DATA_PATH
    )
    with init_devices(mock=True):
        detector = EigerDetector("MOCK:EIGER:", path_provider, name="test_eiger")
    set_mock_value(detector.fileio.file_path_exists, True)
    set_mock_value(detector.driver.array_size_x, 2048)
    set_mock_value(detector.driver.array_size_y, 2048)
    set_mock_value(detector.driver.data_type, "UInt16")

    yield detector

    if EIGER_DATA_PATH.exists():
        shutil.rmtree(EIGER_DATA_PATH)


@pytest.fixture
def mock_eiger_driver(RE: RunEngine) -> EigerDriverIO:
    """Create a mock EigerDriverIO for testing."""
    with init_devices(mock=True):
        driver = EigerDriverIO("MOCK:EIGER:cam1:")

    # Set up some default mock values
    set_mock_value(driver.file_path_exists, True)
    set_mock_value(driver.array_size_x, 2048)
    set_mock_value(driver.array_size_y, 2048)
    set_mock_value(driver.data_type, ADBaseDataType.UINT16)

    return driver


@pytest.fixture
def mock_path_provider() -> PathProvider:
    """Create a mock path provider for testing."""
    return StaticPathProvider(
        StaticFilenameProvider("test_eiger"), directory_path=EIGER_DATA_PATH
    )


@pytest.fixture
def eiger_writer(
    mock_eiger_driver: EigerDriverIO,
    mock_path_provider: PathProvider,
) -> Generator[EigerWriter, None, None]:
    """Create an EigerWriter instance for testing."""
    if not EIGER_DATA_PATH.exists():
        EIGER_DATA_PATH.mkdir(parents=True)
    assert EIGER_DATA_PATH.exists()
    dataset_describer = ADBaseDatasetDescriber(mock_eiger_driver)
    yield EigerWriter(mock_eiger_driver, mock_path_provider, dataset_describer)
    if EIGER_DATA_PATH.exists():
        shutil.rmtree(EIGER_DATA_PATH)


@pytest.fixture
def eiger_controller(mock_eiger_driver: EigerDriverIO) -> EigerController:
    return EigerController(mock_eiger_driver)


@pytest.mark.asyncio
async def test_eiger_writer_initialization(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
    mock_path_provider: PathProvider,
):
    """Test that EigerWriter initializes correctly."""
    assert eiger_writer.fileio is mock_eiger_driver
    assert eiger_writer._path_provider is mock_path_provider  # type: ignore[reportPrivateUsage]
    assert eiger_writer._dataset_describer is not None  # type: ignore[reportPrivateUsage]
    assert eiger_writer._file_info is None  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_eiger_writer_open(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
) -> None:
    """Test the open method configures the detector correctly."""
    array_size_x, array_size_y = await asyncio.gather(
        mock_eiger_driver.array_size_x.get_value(),
        mock_eiger_driver.array_size_y.get_value(),
    )

    # Case 1: 1 image, 1 trigger
    set_mock_value(mock_eiger_driver.sequence_id, 0)
    set_mock_value(mock_eiger_driver.num_images, 1)

    description = await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    assert await mock_eiger_driver.fw_enable.get_value() is True
    assert await mock_eiger_driver.save_files.get_value() is True
    assert description.keys() == {
        # TODO: Add these when empty shape datasets are supported by tiled
        # "test_eiger_y_pixel_size",
        # "test_eiger_x_pixel_size",
        # "test_eiger_detector_distance",
        # "test_eiger_incident_wavelength",
        # "test_eiger_frame_time",
        # "test_eiger_beam_center_x",
        # "test_eiger_beam_center_y",
        # "test_eiger_count_time",
        # "test_eiger_pixel_mask",
        "test_eiger_image",
    }
    assert description["test_eiger_image"]["source"] == "ADEiger FileWriter"

    # Case 2: 4 images per file, 11 images, 2 triggers
    # Expect 6 files, the first 5 will have 4 images, the last will have 2
    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.num_images, 11)
    description = await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert description.keys() == {
        # TODO: Add these when empty shape datasets are supported by tiled
        # "test_eiger_y_pixel_size",
        # "test_eiger_x_pixel_size",
        # "test_eiger_detector_distance",
        # "test_eiger_incident_wavelength",
        # "test_eiger_frame_time",
        # "test_eiger_beam_center_x",
        # "test_eiger_beam_center_y",
        # "test_eiger_count_time",
        # "test_eiger_pixel_mask",
        "test_eiger_image",
    }
    data_key = description["test_eiger_image"]
    assert tuple(data_key["shape"]) == (11, array_size_x, array_size_y)
    assert data_key["dtype"] == "array"
    assert "dtype_numpy" in data_key
    assert data_key["dtype_numpy"] == np.dtype(np.uint32).str
    assert "external" in data_key
    assert data_key["external"] == "STREAM:"
    assert data_key["source"] == "ADEiger FileWriter"


@pytest.mark.asyncio
async def test_eiger_writer_get_indices_written(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
):
    """Test getting the number of indices written."""
    set_mock_value(mock_eiger_driver.sequence_id, 1)

    # Case 1: 1 image, 1 trigger
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.array_counter, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.array_counter, 1)
    assert await eiger_writer.get_indices_written() == 1

    # Case 2: 1 image, 5 triggers
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.array_counter, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.array_counter, 1)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_driver.array_counter, 3)
    assert await eiger_writer.get_indices_written() == 3
    set_mock_value(mock_eiger_driver.array_counter, 5)
    assert await eiger_writer.get_indices_written() == 5

    # Case 3: 5 images, 2 triggers
    set_mock_value(mock_eiger_driver.num_images, 5)
    set_mock_value(mock_eiger_driver.array_counter, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value(),
    )
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.array_counter, 4)
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.array_counter, 5)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_driver.array_counter, 9)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_driver.array_counter, 10)
    assert await eiger_writer.get_indices_written() == 2


@pytest.mark.asyncio
async def test_eiger_writer_observe_indices_written(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
) -> None:
    """Test observing indices as they are written."""
    set_mock_value(mock_eiger_driver.sequence_id, 1)

    async def _simulate_writing_indices(
        num_images: int, num_triggers: int, acquire_time: float = 0.01
    ) -> list[int]:
        # Create an async generator to track yielded indices
        observed_indices = []
        set_mock_value(mock_eiger_driver.array_counter, 0)

        async def _simulate_acquisition():
            """Simulate the detector writing images by incrementing sequence_id."""
            for i in range(1, num_images * num_triggers + 1):
                await asyncio.sleep(acquire_time)
                set_mock_value(mock_eiger_driver.array_counter, i)

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
) -> None:
    """Test collecting stream documents."""

    async def collect_docs(
        num_triggers: int,
    ) -> tuple[list[StreamResource], list[StreamDatum]]:
        resource_docs = []
        data_docs = []
        for i in range(1, num_triggers + 1):
            sequence_id = await mock_eiger_driver.sequence_id.get_value()
            set_mock_value(mock_eiger_driver.sequence_id, sequence_id + 1)
            async for doc_type, doc in eiger_writer.collect_stream_docs(
                name="", indices_written=i
            ):
                if doc_type == "stream_resource":
                    resource_docs.append(doc)
                elif doc_type == "stream_datum":
                    data_docs.append(doc)
        return resource_docs, data_docs

    set_mock_value(mock_eiger_driver.sequence_id, 0)
    set_mock_value(mock_eiger_driver.num_images, 1)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    resource_docs, data_docs = await collect_docs(num_triggers=1)
    assert len(resource_docs) == 1
    assert len(data_docs) == 1
    assert (
        resource_docs[0]["uri"]
        == f"file://localhost{EIGER_DATA_PATH}/test_eiger_1_master.h5"
    )

    await eiger_writer.close()

    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    resource_docs, data_docs = await collect_docs(num_triggers=3)
    assert len(resource_docs) == 3
    assert len(data_docs) == 3
    # There are 10 different datasets inside a single master file
    # 3 triggers, so 30 total resources/datasets
    assert (
        resource_docs[0]["uri"]
        == f"file://localhost{EIGER_DATA_PATH}/test_eiger_2_master.h5"
    )
    assert (
        resource_docs[1]["uri"]
        == f"file://localhost{EIGER_DATA_PATH}/test_eiger_3_master.h5"
    )
    assert (
        resource_docs[2]["uri"]
        == f"file://localhost{EIGER_DATA_PATH}/test_eiger_4_master.h5"
    )


@pytest.mark.asyncio
async def test_eiger_writer_close(
    eiger_writer: EigerWriter,
    mock_eiger_driver: EigerDriverIO,
) -> None:
    """Test closing the writer."""

    # Verify the writing was enabled
    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    assert await mock_eiger_driver.fw_enable.get_value() is True
    assert await mock_eiger_driver.save_files.get_value() is True

    # Verify the writing was disabled
    await eiger_writer.close()
    assert eiger_writer._file_info is None  # type: ignore[reportPrivateUsage]


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
    assert await eiger_controller.driver.num_images.get_value() == 5
    assert await eiger_controller.driver.image_mode.get_value() == ADImageMode.MULTIPLE

    trigger_info = TriggerInfo(
        number_of_events=0,
        livetime=None,
        deadtime=0.0,
        trigger=DetectorTrigger.EDGE_TRIGGER,
        exposure_timeout=10.0,
        exposures_per_event=1,
    )
    await eiger_controller.prepare(trigger_info)
    assert await eiger_controller.driver.acquire_time.get_value() == 0.0
    assert (
        await eiger_controller.driver.trigger_mode.get_value()
        == EigerTriggerMode.EXTERNAL_SERIES
    )
    assert await eiger_controller.driver.num_images.get_value() == 1
    assert (
        await eiger_controller.driver.image_mode.get_value() == ADImageMode.CONTINUOUS
    )


@pytest.mark.asyncio
async def test_eiger_detector(mock_eiger_detector: EigerDetector) -> None:
    set_mock_value(mock_eiger_detector.driver.num_images, 1)
    set_mock_value(mock_eiger_detector.driver.acquire_period, 0.001)
    set_mock_value(mock_eiger_detector.fileio.array_counter, 0)

    async def _simulate_one_trigger(value: bool, wait: bool) -> None:
        await asyncio.sleep(await mock_eiger_detector.driver.acquire_period.get_value())
        array_counter = await mock_eiger_detector.fileio.array_counter.get_value()
        set_mock_value(mock_eiger_detector.fileio.array_counter, array_counter + 1)

    callback_on_mock_put(mock_eiger_detector.driver.acquire, _simulate_one_trigger)

    # Standalone methods
    await mock_eiger_detector.describe()

    # Case 1 - Step Scan: stage, trigger, read, trigger, read, unstage
    await mock_eiger_detector.stage()
    await mock_eiger_detector.trigger()
    assert (
        await mock_eiger_detector.driver.data_source.get_value()
        == EigerDataSource.FILE_WRITER
    )
    await mock_eiger_detector.read()
    await mock_eiger_detector.trigger()
    await mock_eiger_detector.read()
    await mock_eiger_detector.unstage()

    set_mock_value(mock_eiger_detector.fileio.array_counter, 0)
    # Case 2 - Fly Scan: prepare, kickoff, complete
    await mock_eiger_detector.prepare(
        TriggerInfo(
            number_of_events=1,
            livetime=0.01,
            deadtime=0.001,
            trigger=DetectorTrigger.INTERNAL,
            exposure_timeout=10.0,
            exposures_per_event=1,
        )
    )
    await mock_eiger_detector.kickoff()
    await mock_eiger_detector.complete()


@pytest.mark.asyncio
async def test_eiger_detector_with_RE(
    RE: RunEngine, tiled_client: Container, mock_eiger_detector: EigerDetector
) -> None:
    RE.subscribe(print)
    set_mock_value(mock_eiger_detector.fileio.array_counter, 0)

    async def _write_file(value: bool, wait: bool) -> None:
        if value:
            num_images = await mock_eiger_detector.driver.num_images.get_value()
            sequence_id = await mock_eiger_detector.fileio.sequence_id.get_value() + 1
            set_mock_value(mock_eiger_detector.fileio.sequence_id, sequence_id)
            await asyncio.sleep(
                await mock_eiger_detector.driver.acquire_period.get_value()
            )
            write_eiger_hdf5_file(
                num_images=num_images,
                sequence_id=sequence_id,
                name="test_eiger",
            )
            array_counter = await mock_eiger_detector.fileio.array_counter.get_value()
            set_mock_value(
                mock_eiger_detector.fileio.array_counter, array_counter + num_images
            )

    tiled_writer = TiledWriter(tiled_client)
    RE.subscribe(tiled_writer)
    callback_on_mock_put(mock_eiger_detector.driver.acquire, _write_file)

    set_mock_value(mock_eiger_detector.fileio.sequence_id, 0)
    set_mock_value(mock_eiger_detector.driver.num_images, 1)
    set_mock_value(mock_eiger_detector.driver.acquire_period, 0.001)
    uid = RE(bp.count([mock_eiger_detector]))
    assert uid is not None
    assert (
        tiled_client.values().last()["primary"]["test_eiger_image"].read() is not None
    )
    assert tiled_client.values().last()["primary"]["test_eiger_image"].shape == (
        1,
        2048,
        2048,
    )
    assert (
        tiled_client.values().last()["primary"]["test_eiger_image"].dtype == np.uint32
    )
    # TODO: Add these when empty shape datasets are supported by tiled
    # assert tiled_client.values().last()["primary"]["test_eiger_x_pixel_size"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_x_pixel_size"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_x_pixel_size"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_y_pixel_size"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_y_pixel_size"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_y_pixel_size"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_detector_distance"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_detector_distance"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_detector_distance"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_incident_wavelength"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_incident_wavelength"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_incident_wavelength"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_frame_time"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_frame_time"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_frame_time"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_beam_center_x"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_beam_center_x"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_beam_center_x"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_beam_center_y"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_beam_center_y"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_beam_center_y"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_count_time"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_count_time"
    # ].shape == ()
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_count_time"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_pixel_mask"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_pixel_mask"
    # ].shape == (2048, 2048)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_pixel_mask"]
    #     .dtype
    #     == np.uint32
    # )

    set_mock_value(mock_eiger_detector.fileio.sequence_id, 2)
    set_mock_value(mock_eiger_detector.driver.num_images, 1)
    set_mock_value(mock_eiger_detector.driver.acquire_period, 0.001)
    uid = RE(bp.count([mock_eiger_detector], num=10))
    assert uid is not None
    assert (
        tiled_client.values().last()["primary"]["test_eiger_image"].read() is not None
    )
    assert tiled_client.values().last()["primary"]["test_eiger_image"].shape == (
        10,
        2048,
        2048,
    )
    assert (
        tiled_client.values().last()["primary"]["test_eiger_image"].dtype == np.uint32
    )
    # TODO: Add these when empty shape datasets are supported by tiled
    # assert tiled_client.values().last()["primary"]["test_eiger_x_pixel_size"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_x_pixel_size"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_x_pixel_size"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_y_pixel_size"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_y_pixel_size"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_y_pixel_size"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_detector_distance"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_detector_distance"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_detector_distance"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_incident_wavelength"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_incident_wavelength"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_incident_wavelength"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_frame_time"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_frame_time"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_frame_time"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_beam_center_x"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_beam_center_x"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_beam_center_x"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_beam_center_y"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_beam_center_y"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_beam_center_y"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_count_time"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_count_time"
    # ].shape == (10,)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_count_time"]
    #     .dtype
    #     == np.float32
    # )
    # assert tiled_client.values().last()["primary"]["test_eiger_pixel_mask"].read() is not None
    # assert tiled_client.values().last()["primary"][
    #     "test_eiger_pixel_mask"
    # ].shape == (10, 2048, 2048)
    # assert (
    #     tiled_client.values()
    #     .last()["primary"]["test_eiger_pixel_mask"]
    #     .dtype
    #     == np.uint32
    # )
