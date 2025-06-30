"""
Tests for the EigerWriter class using ophyd-async mocking utilities.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from bluesky.run_engine import RunEngine
import pytest
from ophyd_async.core import (
    init_devices,
    PathInfo,
    StaticPathProvider,
    StaticFilenameProvider,
)
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
    set_mock_value(driver.sequence_id, 0.0)
    set_mock_value(driver.fw_nimgs_per_file, 1000.0)
    set_mock_value(driver.array_size_x, 2048)
    set_mock_value(driver.array_size_y, 2048)
    set_mock_value(driver.data_type, "UInt16")
    
    return driver


@pytest.fixture
def mock_path_provider():
    """Create a mock path provider for testing."""
    return StaticPathProvider(
        StaticFilenameProvider("test_eiger"),
        directory_path="/tmp/test_data"
    )


@pytest.fixture
def eiger_writer(mock_eiger_driver, mock_path_provider) -> EigerWriter:
    """Create an EigerWriter instance for testing."""
    return EigerWriter(mock_eiger_driver, mock_path_provider)


@pytest.mark.asyncio
async def test_eiger_writer_initialization(eiger_writer, mock_eiger_driver, mock_path_provider):
    """Test that EigerWriter initializes correctly."""
    assert eiger_writer._driver is mock_eiger_driver
    assert eiger_writer._path_provider is mock_path_provider
    assert eiger_writer._sequence_id_offset == 1


@pytest.mark.asyncio
async def test_eiger_writer_open(eiger_writer, mock_eiger_driver):
    """Test the open method configures the detector correctly."""
    # Mock the file path creation
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        describe_doc = await eiger_writer.open(multiplier=1)
    
    # Check that the describe document has the expected structure
    assert "primary" in describe_doc
    primary_doc = describe_doc["primary"]
    
    assert "source" in primary_doc
    assert primary_doc["source"].startswith("EIGER:")
    assert primary_doc["shape"] == [2048, 2048]
    assert primary_doc["dtype"] == "uint16"
    assert "FILESTORE:" in primary_doc["external"]
    
    # Verify the detector was configured correctly
    assert await mock_eiger_driver.fw_enable.get_value() is True
    assert await mock_eiger_driver.save_files.get_value() is True
    assert await mock_eiger_driver.fw_hdf5_format.get_value() == EigerHDF5Format.V2024_2
    assert await mock_eiger_driver.fw_compression.get_value() is True
    assert await mock_eiger_driver.data_source.get_value() == EigerDataSource.FILE_WRITER


@pytest.mark.asyncio
async def test_eiger_writer_get_indices_written(eiger_writer, mock_eiger_driver):
    """Test getting the number of indices written."""
    # Initialize the writer first
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        await eiger_writer.open()
    
    # Initial state should be 0
    assert await eiger_writer.get_indices_written() == 0
    
    # Simulate some images being written
    set_mock_value(mock_eiger_driver.sequence_id, 3.0)
    assert await eiger_writer.get_indices_written() == 3


@pytest.mark.asyncio
async def test_eiger_writer_observe_indices_written(eiger_writer, mock_eiger_driver):
    """Test observing indices as they are written."""
    # Initialize the writer first
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        await eiger_writer.open()
    
    # Create an async generator to track yielded indices
    observed_indices = []
    
    async def simulate_writing():
        """Simulate the detector writing images by incrementing sequence_id."""
        await asyncio.sleep(0.05)  # Small delay
        set_mock_value(mock_eiger_driver.sequence_id, 1.0)
        await asyncio.sleep(0.05)
        set_mock_value(mock_eiger_driver.sequence_id, 2.0)
        await asyncio.sleep(0.05)
        set_mock_value(mock_eiger_driver.sequence_id, 3.0)
    
    # Start the simulation task
    sim_task = asyncio.create_task(simulate_writing())
    
    # Observe the indices being written
    observe_task = asyncio.create_task(
        _collect_observed_indices(eiger_writer, observed_indices, max_indices=3)
    )
    
    # Wait for both tasks to complete
    await asyncio.gather(sim_task, observe_task)
    
    # Should have observed indices 0, 1, 2
    assert observed_indices == [0, 1, 2]


async def _collect_observed_indices(writer, indices_list, max_indices):
    """Helper function to collect observed indices."""
    count = 0
    async for index in writer.observe_indices_written(timeout=1.0):
        indices_list.append(index)
        count += 1
        if count >= max_indices:
            break


@pytest.mark.asyncio
async def test_eiger_writer_collect_stream_docs(eiger_writer, mock_eiger_driver):
    """Test collecting stream documents."""
    # Initialize the writer first
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        await eiger_writer.open()
    
    # Collect stream docs for 2 indices
    docs = []
    async for doc_type, doc in eiger_writer.collect_stream_docs(indices_written=2):
        docs.append((doc_type, doc))
    
    # Should have 1 stream_resource + 2 stream_datum docs
    assert len(docs) == 3
    
    # Check stream_resource document
    assert docs[0][0] == "stream_resource"
    stream_resource = docs[0][1]
    assert stream_resource["spec"] == "AD_EIGER"
    assert stream_resource["root"] == "/tmp/test_data"
    assert "resource_path" in stream_resource
    assert stream_resource["resource_kwargs"]["images_per_file"] == 1000
    
    # Check stream_datum documents
    for i in range(1, 3):
        assert docs[i][0] == "stream_datum"
        stream_datum = docs[i][1]
        assert stream_datum["descriptor"] == "primary"
        assert stream_datum["stream_resource"] == stream_resource["uid"]
        assert stream_datum["seq_num"] == i - 1
        assert "datum_kwargs" in stream_datum
        assert "seq_id" in stream_datum["datum_kwargs"]


@pytest.mark.asyncio
async def test_eiger_writer_close(eiger_writer, mock_eiger_driver):
    """Test closing the writer."""
    # Initialize the writer first
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        await eiger_writer.open()
    
    # Mock file existence check
    with patch("pathlib.Path.exists", return_value=True):
        await eiger_writer.close()
    
    # Verify the detector was disabled
    assert await mock_eiger_driver.fw_enable.get_value() is False
    assert await mock_eiger_driver.save_files.get_value() is False


@pytest.mark.asyncio
async def test_eiger_writer_data_type_mapping(eiger_writer, mock_eiger_driver):
    """Test that different data types are mapped correctly."""
    data_type_tests = [
        ("UInt8", "uint8"),
        ("UInt16", "uint16"),
        ("UInt32", "uint32"),
        ("Int8", "int8"),
        ("Int16", "int16"),
        ("Int32", "int32"),
        ("Float32", "float32"),
        ("Float64", "float64"),
        ("Unknown", "uint16"),  # Should default to uint16
    ]
    
    for eiger_type, expected_numpy_type in data_type_tests:
        set_mock_value(mock_eiger_driver.data_type, eiger_type)
        
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
            describe_doc = await eiger_writer.open()
        
        assert describe_doc["primary"]["dtype"] == expected_numpy_type


@pytest.mark.asyncio
async def test_eiger_writer_file_validation_warning(eiger_writer, mock_eiger_driver, caplog):
    """Test that missing files generate warnings."""
    # Initialize the writer first
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        await eiger_writer.open()
    
    # Simulate some images being written
    set_mock_value(mock_eiger_driver.sequence_id, 2.0)
    
    # Mock file existence check to return False (files don't exist)
    with patch("pathlib.Path.exists", return_value=False):
        await eiger_writer.close()
    
    # Check that a warning was logged about missing files
    assert "Master files were not written" in caplog.text


@pytest.mark.asyncio
async def test_eiger_writer_resource_uid_generation(eiger_writer):
    """Test that resource UIDs are generated consistently."""
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
        describe_doc1 = await eiger_writer.open()
        
        # Resource UID should be set and be 8 characters long
        assert eiger_writer._resource_uid is not None
        assert len(eiger_writer._resource_uid) == 8
        
        # Source should include the resource UID
        assert eiger_writer._resource_uid in describe_doc1["primary"]["source"]


@pytest.mark.asyncio
async def test_eiger_writer_path_creation(eiger_writer):
    """Test that write paths are created correctly."""
    with patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("pathlib.Path.exists", return_value=False):
        
        # Mock datetime to have consistent path
        with patch("cditools.eiger_async.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 12, 25, 10, 30, 0)
            mock_datetime.strftime = datetime.strftime
            
            await eiger_writer.open()
        
        # Check that the path was created
        mock_mkdir.assert_called_once()
        expected_path = Path("/tmp/test_data/2023/12/25")
        assert eiger_writer._write_path == expected_path 