"""Tests for pandas DataFrame codecs."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from tinman import ObLogWriter, ObLogReader
from tinman.codecs import DataFrameCodec


class TestDataFrameCodec:
    """Test DataFrameCodec (zero-copy) functionality."""

    def test_encode_decode_numeric(self):
        """Test encoding and decoding numeric-only DataFrames."""
        codec = DataFrameCodec()
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0]
        })
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        pd.testing.assert_frame_equal(df, decoded)

    def test_encode_decode_various_dtypes(self):
        """Test different numpy dtypes in DataFrames."""
        codec = DataFrameCodec()
        
        df = pd.DataFrame({
            'int8': np.array([1, 2, 3], dtype=np.int8),
            'int32': np.array([10, 20, 30], dtype=np.int32),
            'int64': np.array([100, 200, 300], dtype=np.int64),
            'float32': np.array([1.1, 2.2, 3.3], dtype=np.float32),
            'float64': np.array([10.1, 20.2, 30.3], dtype=np.float64),
            'bool': np.array([True, False, True], dtype=np.bool_),
        })
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        pd.testing.assert_frame_equal(df, decoded)
        
        # Verify dtypes preserved
        for col in df.columns:
            assert df[col].dtype == decoded[col].dtype

    def test_large_dataframe(self):
        """Test zero-copy codec with large DataFrames."""
        codec = DataFrameCodec()
        
        df = pd.DataFrame({
            'x': np.random.rand(10000),
            'y': np.random.rand(10000),
            'z': np.random.randint(0, 100, 10000),
        })
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        pd.testing.assert_frame_equal(df, decoded)

    async def test_oblog_integration(self):
        """Test DataFrameCodec with ObLog."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Write
            async with ObLogWriter(log_dir) as oblog:
                write = oblog.get_writer("metrics", DataFrameCodec())
                
                df1 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
                df2 = pd.DataFrame({'a': [5, 6], 'b': [7.0, 8.0]})
                
                write(df1)
                write(df2)
            
            # Read
            reader = ObLogReader(log_dir)
            results = []
            async for timestamp, df in reader.read_channel("metrics"):
                results.append(df)
            
            assert len(results) == 2
            pd.testing.assert_frame_equal(results[0], df1)
            pd.testing.assert_frame_equal(results[1], df2)

    def test_empty_dataframe(self):
        """Test encoding/decoding empty DataFrames."""
        codec = DataFrameCodec()
        df = pd.DataFrame({'a': pd.Series([], dtype=np.int64), 'b': pd.Series([], dtype=np.float64)})
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        pd.testing.assert_frame_equal(df, decoded)

    def test_single_row_dataframe(self):
        """Test encoding/decoding single-row DataFrames."""
        codec = DataFrameCodec()
        df = pd.DataFrame({'a': [42], 'b': [3.14]})
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        pd.testing.assert_frame_equal(df, decoded)

    def test_many_columns(self):
        """Test DataFrames with many columns."""
        codec = DataFrameCodec()
        
        # Create DataFrame with 100 columns
        data = {f'col_{i}': [i, i+1, i+2] for i in range(100)}
        df = pd.DataFrame(data)
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        pd.testing.assert_frame_equal(df, decoded)

    async def test_zero_copy_view(self):
        """Verify decoded DataFrames have zero-copy columns."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Write
            async with ObLogWriter(log_dir) as oblog:
                write = oblog.get_writer("data", DataFrameCodec())
                df = pd.DataFrame({
                    'x': np.array([1.0, 2.0, 3.0], dtype=np.float64),
                    'y': np.array([4, 5, 6], dtype=np.int32)
                })
                write(df)
            
            # Read
            reader = ObLogReader(log_dir)
            async for _, df in reader.read_channel("data"):
                # Numeric columns should be views
                for col in df.columns:
                    arr = df[col].values
                    assert arr.base is not None, f"Column '{col}' should be a view"
                    if hasattr(arr, 'flags'):
                        assert not arr.flags.writeable, f"Column '{col}' should be read-only"

    def test_data_integrity(self):
        """Test that decoded data is bit-for-bit identical."""
        codec = DataFrameCodec()
        
        # Use various data patterns
        df = pd.DataFrame({
            'zeros': [0.0, 0.0, 0.0],
            'ones': [1.0, 1.0, 1.0],
            'negatives': [-1.5, -2.5, -3.5],
            'large': [1e10, 2e10, 3e10],
            'small': [1e-10, 2e-10, 3e-10],
        })
        
        encoded = codec.encode(df)
        decoded = codec.decode(encoded)
        
        # Check exact equality (not just approximate)
        pd.testing.assert_frame_equal(df, decoded, check_exact=True)
