"""
Tests for the EigerWriter class using ophyd-async mocking utilities.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from bluesky.run_engine import RunEngine
import pytest
from ophyd_async.core import (
    PathProvider,
    init_devices,
    PathInfo,
    StaticPathProvider,
    StaticFilenameProvider,
)
from ophyd_async.epics.adcore import ADBaseDatasetDescriber
from ophyd_async.testing import (
    set_mock_value,
    assert_reading,
    assert_value,
)

from cditools.eiger_async import (
    EigerDriverIO,
    EigerWriter,
    EigerDataSource,
    EigerHDF5Format,
)


@pytest.fixture
def mock_eiger_driver(RE: RunEngine) -> EigerDriverIO:
    """Create a mock EigerDriverIO for testing."""
    with init_devices(mock=True):
        driver = EigerDriverIO("MOCK:EIGER:CAM:")
    
    # Set up some default mock values
    set_mock_value(driver.array_size_x, 2048)
    set_mock_value(driver.array_size_y, 2048)
    set_mock_value(driver.data_type, "UInt16")
    
    return driver


@pytest.fixture
def mock_path_provider() -> PathProvider:
    """Create a mock path provider for testing."""
    return StaticPathProvider(
        StaticFilenameProvider("test_eiger"),
        directory_path="/tmp/test_data"
    )


@pytest.fixture
def eiger_writer(mock_eiger_driver: EigerDriverIO, mock_path_provider: PathProvider) -> EigerWriter:
    """Create an EigerWriter instance for testing."""
    dataset_describer = ADBaseDatasetDescriber(mock_eiger_driver)
    return EigerWriter(mock_eiger_driver, mock_path_provider, dataset_describer)


@pytest.mark.asyncio
async def test_eiger_writer_initialization(eiger_writer: EigerWriter, mock_eiger_driver: EigerDriverIO, mock_path_provider: PathProvider):
    """Test that EigerWriter initializes correctly."""
    assert eiger_writer._driver is mock_eiger_driver
    assert eiger_writer._path_provider is mock_path_provider
    assert eiger_writer._dataset_describer is not None
    assert eiger_writer._file_info is None
    assert eiger_writer._current_sequence_id is None
    assert eiger_writer._composer is None


@pytest.mark.asyncio
async def test_eiger_writer_open(eiger_writer: EigerWriter, mock_eiger_driver: EigerDriverIO) -> None:
    """Test the open method configures the detector correctly."""
    array_size_x, array_size_y, data_type = await asyncio.gather(
        mock_eiger_driver.array_size_x.get_value(),
        mock_eiger_driver.array_size_y.get_value(),
        mock_eiger_driver.data_type.get_value(),
    )

    # Case 1: 1 image per file, 1 image, 1 trigger
    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.fw_nimgs_per_file, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)

    description = await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    assert await mock_eiger_driver.fw_enable.get_value() is True
    assert await mock_eiger_driver.save_files.get_value() is True
    assert await mock_eiger_driver.fw_hdf5_format.get_value() == EigerHDF5Format.LEGACY
    assert await mock_eiger_driver.data_source.get_value() == EigerDataSource.FILE_WRITER
    assert description.keys() == {"test_eiger_y_pixel_size",
                                  "test_eiger_x_pixel_size",
                                  "test_eiger_detector_distance",
                                  "test_eiger_incident_wavelength",
                                  "test_eiger_frame_time",
                                  "test_eiger_beam_center_x",
                                  "test_eiger_beam_center_y",
                                  "test_eiger_count_time",
                                  "test_eiger_pixel_mask",
                                  "test_eiger_1"}
    assert description["test_eiger_1"]["source"] == "/tmp/test_data/test_eiger_1_master.h5"

    # Case 2: 4 images per file, 11 images, 2 triggers
    # Expect 6 files, the first 5 will have 4 images, the last will have 2
    set_mock_value(mock_eiger_driver.sequence_id, 2)
    set_mock_value(mock_eiger_driver.fw_nimgs_per_file, 4)
    set_mock_value(mock_eiger_driver.num_images, 11)
    set_mock_value(mock_eiger_driver.num_triggers, 2)
    description = await eiger_writer.open(name="test_eiger", exposures_per_event=await mock_eiger_driver.num_images.get_value())
    assert description.keys() == {"test_eiger_y_pixel_size",
                                  "test_eiger_x_pixel_size",
                                  "test_eiger_detector_distance",
                                  "test_eiger_incident_wavelength",
                                  "test_eiger_frame_time",
                                  "test_eiger_beam_center_x",
                                  "test_eiger_beam_center_y",
                                  "test_eiger_count_time",
                                  "test_eiger_pixel_mask",
                                  "test_eiger_1",
                                  "test_eiger_2",
                                  "test_eiger_3",
                                  "test_eiger_4",
                                  "test_eiger_5",
                                  "test_eiger_6"}

    for i in range(1, 6):
        data_key = description[f"test_eiger_{i}"]
        assert tuple(data_key["shape"]) == (4, array_size_x, array_size_y)
        assert data_key["dtype"] == "array"
        assert data_key["dtype_numpy"] == np.dtype(data_type.lower()).str
        assert data_key["external"] == "STREAM:"
        assert data_key["source"] == f"/tmp/test_data/test_eiger_2_master.h5"
    data_key = description["test_eiger_6"]
    assert tuple(data_key["shape"]) == (2, array_size_x, array_size_y)
    assert data_key["dtype"] == "array"
    assert data_key["dtype_numpy"] == np.dtype(data_type.lower()).str
    assert data_key["external"] == "STREAM:"
    assert data_key["source"] == f"/tmp/test_data/test_eiger_2_master.h5"


@pytest.mark.asyncio
async def test_eiger_writer_get_indices_written(eiger_writer, mock_eiger_driver):
    """Test getting the number of indices written."""
    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.fw_nimgs_per_file, 1)

    # Case 1: 1 image, 1 trigger
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_images_counter, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value())
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.num_images_counter, 1)
    assert await eiger_writer.get_indices_written() == 1

    # Case 2: 1 image, 5 triggers
    set_mock_value(mock_eiger_driver.num_triggers, 5)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_images_counter, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value())
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.num_images_counter, 1)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_driver.num_images_counter, 3)
    assert await eiger_writer.get_indices_written() == 3
    set_mock_value(mock_eiger_driver.num_images_counter, 5)
    assert await eiger_writer.get_indices_written() == 5

    # Case 3: 5 images, 2 triggers
    set_mock_value(mock_eiger_driver.num_triggers, 2)
    set_mock_value(mock_eiger_driver.num_images, 5)
    set_mock_value(mock_eiger_driver.num_images_counter, 0)
    await eiger_writer.open(
        name="test_eiger",
        exposures_per_event=await mock_eiger_driver.num_images.get_value())
    assert await eiger_writer.get_indices_written() == 0
    set_mock_value(mock_eiger_driver.num_images_counter, 4)
    assert await eiger_writer.get_indices_written() == 0 
    set_mock_value(mock_eiger_driver.num_images_counter, 5)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_driver.num_images_counter, 9)
    assert await eiger_writer.get_indices_written() == 1
    set_mock_value(mock_eiger_driver.num_images_counter, 10)
    assert await eiger_writer.get_indices_written() == 2



@pytest.mark.asyncio
async def test_eiger_writer_observe_indices_written(eiger_writer: EigerWriter, mock_eiger_driver: EigerDriverIO) -> None:
    """Test observing indices as they are written."""
    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.fw_nimgs_per_file, 1)

    async def _simulate_writing_indices(num_images: int, num_triggers: int, acquire_time: float = 0.01) -> list[int]:
        # Create an async generator to track yielded indices
        observed_indices = []
        set_mock_value(mock_eiger_driver.num_images_counter, 0)
        
        async def _simulate_acquisition():
            """Simulate the detector writing images by incrementing sequence_id."""
            for i in range(1, num_images * num_triggers + 1):
                await asyncio.sleep(acquire_time)
                set_mock_value(mock_eiger_driver.num_images_counter, i)

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
    observed = await _simulate_writing_indices(num_images=num_images, num_triggers=num_triggers)
    assert observed == [0, 1]

    # Case 2: 1 image, 5 triggers
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 5)
    num_images = await mock_eiger_driver.num_images.get_value()
    num_triggers = await mock_eiger_driver.num_triggers.get_value()
    await eiger_writer.open(name="test_eiger", exposures_per_event=num_images)
    observed = await _simulate_writing_indices(num_images=num_images, num_triggers=num_triggers)
    assert observed == [0, 1, 2, 3, 4, 5]

    # Case 3: 5 images, 2 triggers
    set_mock_value(mock_eiger_driver.num_images, 5)
    set_mock_value(mock_eiger_driver.num_triggers, 2)
    num_images = await mock_eiger_driver.num_images.get_value()
    num_triggers = await mock_eiger_driver.num_triggers.get_value()
    await eiger_writer.open(name="test_eiger", exposures_per_event=num_images)
    observed = await _simulate_writing_indices(num_images=num_images, num_triggers=num_triggers)
    assert observed == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]


@pytest.mark.asyncio
async def test_eiger_writer_collect_stream_docs(eiger_writer: EigerWriter, mock_eiger_driver: EigerDriverIO) -> None:
    """Test collecting stream documents."""

    async def collect_docs(num_triggers: int):
        resource_docs = []
        data_docs = []
        for i in range(1, num_triggers + 1):
            async for doc_type, doc in eiger_writer.collect_stream_docs(name="test_eiger", indices_written=i):
                if doc_type == "stream_resource":
                    resource_docs.append(doc)
                elif doc_type == "stream_datum":
                    data_docs.append(doc)
        return resource_docs, data_docs

    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.fw_nimgs_per_file, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    resource_docs, data_docs = await collect_docs(num_triggers=await mock_eiger_driver.num_triggers.get_value())
    assert len(resource_docs) == 10
    assert len(data_docs) == 10
    assert resource_docs[0]["uri"] == "file://localhost/tmp/test_data/test_eiger_1_master.h5"

    await eiger_writer.close()

    set_mock_value(mock_eiger_driver.num_triggers, 3)
    set_mock_value(mock_eiger_driver.sequence_id, 2)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    resource_docs, data_docs = await collect_docs(num_triggers=await mock_eiger_driver.num_triggers.get_value())
    assert len(resource_docs) == 12
    assert len(data_docs) == 36
    assert resource_docs[0]["uri"] == "file://localhost/tmp/test_data/test_eiger_2_master.h5"


@pytest.mark.asyncio
async def test_eiger_writer_close(eiger_writer: EigerWriter, mock_eiger_driver: EigerDriverIO) -> None:
    """Test closing the writer."""

    # Verify the writing was enabled
    set_mock_value(mock_eiger_driver.sequence_id, 1)
    set_mock_value(mock_eiger_driver.num_images, 1)
    set_mock_value(mock_eiger_driver.num_triggers, 1)
    set_mock_value(mock_eiger_driver.fw_nimgs_per_file, 1)
    await eiger_writer.open(name="test_eiger", exposures_per_event=1)
    assert await mock_eiger_driver.fw_enable.get_value() is True
    assert await mock_eiger_driver.save_files.get_value() is True
    
    # Verify the writing was disabled
    await eiger_writer.close()
    assert await mock_eiger_driver.fw_enable.get_value() is False
    assert await mock_eiger_driver.save_files.get_value() is False
    assert eiger_writer._composer is None
    assert eiger_writer._current_sequence_id is None
    assert eiger_writer._file_info is None
