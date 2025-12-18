"""Tests for NumpyArrayCodec."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from tinman import ObLogWriter, ObLogReader
from tinman.codecs import NumpyArrayCodec


class TestNumpyArrayCodec:
    """Test NumpyArrayCodec functionality."""

    def test_encode_decode_1d(self):
        """Test encoding and decoding 1D arrays."""
        codec = NumpyArrayCodec()
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        
        encoded = codec.encode(arr)
        decoded = codec.decode(encoded)
        
        np.testing.assert_array_equal(arr, decoded)
        assert decoded.dtype == arr.dtype
        assert decoded.shape == arr.shape

    def test_encode_decode_2d(self):
        """Test encoding and decoding 2D arrays."""
        codec = NumpyArrayCodec()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        encoded = codec.encode(arr)
        decoded = codec.decode(encoded)
        
        np.testing.assert_array_equal(arr, decoded)
        assert decoded.dtype == arr.dtype
        assert decoded.shape == arr.shape

    def test_encode_decode_various_dtypes(self):
        """Test different numpy dtypes."""
        codec = NumpyArrayCodec()
        
        test_cases = [
            np.array([1, 2, 3], dtype=np.int8),
            np.array([1, 2, 3], dtype=np.int16),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([True, False, True], dtype=np.bool_),
        ]
        
        for arr in test_cases:
            encoded = codec.encode(arr)
            decoded = codec.decode(encoded)
            np.testing.assert_array_equal(arr, decoded)
            assert decoded.dtype == arr.dtype

    async def test_oblog_integration(self):
        """Test NumpyArrayCodec with ObLog."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Write arrays
            async with ObLogWriter(log_dir) as oblog:
                write = oblog.get_writer("arrays", NumpyArrayCodec())
                
                arrays = [
                    np.array([1, 2, 3], dtype=np.int32),
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                ]
                
                for arr in arrays:
                    write(arr)
            
            # Read arrays (no context manager needed)
            reader = ObLogReader(log_dir)
            read_arrays = []
            async for _, arr in reader.read_channel("arrays"):
                read_arrays.append(arr.copy())  # Copy to verify
            
            # Verify
            assert len(read_arrays) == len(arrays)
            for orig, read in zip(arrays, read_arrays):
                np.testing.assert_array_equal(orig, read)

    async def test_zero_copy_view(self):
        """Verify decoded arrays are views."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Write
            async with ObLogWriter(log_dir) as oblog:
                write = oblog.get_writer("arrays", NumpyArrayCodec())
                write(np.array([1, 2, 3], dtype=np.int32))
            
            # Read
            reader = ObLogReader(log_dir)
            async for _, arr in reader.read_channel("arrays"):
                # Should be a view
                assert arr.base is not None
                # Should be read-only
                assert not arr.flags.writeable

    def test_empty_array(self):
        """Test encoding/decoding empty arrays."""
        codec = NumpyArrayCodec()
        arr = np.array([], dtype=np.float64)
        
        encoded = codec.encode(arr)
        decoded = codec.decode(encoded)
        
        np.testing.assert_array_equal(arr, decoded)
        assert decoded.dtype == arr.dtype
        assert decoded.shape == arr.shape

    def test_large_array(self):
        """Test encoding/decoding large arrays."""
        codec = NumpyArrayCodec()
        arr = np.random.rand(1000, 1000).astype(np.float32)
        
        encoded = codec.encode(arr)
        decoded = codec.decode(encoded)
        
        np.testing.assert_array_equal(arr, decoded)
        assert decoded.dtype == arr.dtype
        assert decoded.shape == arr.shape
