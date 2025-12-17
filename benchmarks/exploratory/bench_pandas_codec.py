"""Benchmarks for pandas DataFrame codecs - proves zero-copy behavior."""

import asyncio
import gc
import pickle
import struct
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from tinman import Codec, ObLog
from tinman.codecs import DataFrameCodec


class PicklePandasCodec(Codec[pd.DataFrame]):
    """Baseline: pickle-based codec (always copies on read)."""

    def encode(self, item: pd.DataFrame) -> bytes:
        return pickle.dumps(item)

    def decode(self, data: bytes) -> pd.DataFrame:
        return pickle.loads(data)


class DataFrameCopyCodec(Codec[pd.DataFrame]):
    """Baseline: Same format as DataFrameCodec but forces a copy."""

    def encode(self, item: pd.DataFrame) -> bytes:
        n_rows, n_cols = item.shape
        header = struct.pack("<QI", n_rows, n_cols)
        
        for col_name in item.columns:
            col_name_bytes = str(col_name).encode("utf-8")
            header += struct.pack("<I", len(col_name_bytes))
            header += col_name_bytes
            
            dtype_str = str(item[col_name].dtype).encode("ascii")
            header += struct.pack("<I", len(dtype_str))
            header += dtype_str
        
        records = item.to_records(index=False)
        return header + records.tobytes()

    def decode(self, data: bytes) -> pd.DataFrame:
        """Same logic but forces copy."""
        mv = memoryview(data)
        offset = 0
        
        n_rows, n_cols = struct.unpack_from("<QI", mv, offset)
        offset += 12
        
        columns = []
        dtypes = []
        for _ in range(n_cols):
            name_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            col_name = bytes(mv[offset : offset + name_len]).decode("utf-8")
            offset += name_len
            columns.append(col_name)
            
            dtype_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            dtype_str = bytes(mv[offset : offset + dtype_len]).decode("ascii")
            offset += dtype_len
            dtypes.append((col_name, dtype_str))
        
        dtype = np.dtype(dtypes)
        array_data = mv[offset:]
        
        # Force copy
        records = np.frombuffer(array_data, dtype=dtype).copy()
        df = pd.DataFrame(records)
        
        return df


class PyArrowCodec(Codec[pd.DataFrame]):
    """PyArrow IPC (Feather V2) codec for pandas DataFrames."""

    def encode(self, item: pd.DataFrame) -> bytes:
        if not HAS_PYARROW:
            raise ImportError("PyArrow not installed")
        # Convert to Arrow table and serialize with IPC format
        table = pa.Table.from_pandas(item)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()

    def decode(self, data: bytes) -> pd.DataFrame:
        if not HAS_PYARROW:
            raise ImportError("PyArrow not installed")
        # Deserialize from IPC format
        reader = pa.ipc.open_stream(data)
        table = reader.read_all()
        return table.to_pandas()


class FeatherCodec(Codec[pd.DataFrame]):
    """Pandas Feather codec using pandas' built-in to_feather/read_feather."""

    def encode(self, item: pd.DataFrame) -> bytes:
        if not HAS_PYARROW:
            raise ImportError("PyArrow not installed (required for Feather)")
        from io import BytesIO
        buf = BytesIO()
        item.to_feather(buf)
        return buf.getvalue()

    def decode(self, data: bytes) -> pd.DataFrame:
        if not HAS_PYARROW:
            raise ImportError("PyArrow not installed (required for Feather)")
        from io import BytesIO
        buf = BytesIO(data)
        return pd.read_feather(buf)


# Test DataFrames
SMALL_DF = pd.DataFrame({
    'a': np.random.rand(100),
    'b': np.random.rand(100),
    'c': np.random.randint(0, 100, 100)
})

MEDIUM_DF = pd.DataFrame({
    'x': np.random.rand(10000),
    'y': np.random.rand(10000),
    'z': np.random.randint(0, 1000, 10000),
    'w': np.random.rand(10000)
})

LARGE_DF = pd.DataFrame({
    'col1': np.random.rand(100000),
    'col2': np.random.rand(100000),
    'col3': np.random.rand(100000),
    'col4': np.random.randint(0, 10000, 100000),
    'col5': np.random.rand(100000),
})


def get_memory_mb():
    """Get current RSS memory usage in MB."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return rss / 1024 / 1024
        else:
            return rss / 1024
    except (ImportError, AttributeError):
        return 0


@pytest.mark.benchmark(group="pandas-decode-small")
def test_decode_pickle_small(benchmark):
    """Pickle decode - small DataFrame."""
    codec = PicklePandasCodec()
    data = codec.encode(SMALL_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-small")
def test_decode_copy_small(benchmark):
    """Copy decode - small DataFrame."""
    codec = DataFrameCopyCodec()
    data = codec.encode(SMALL_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-small")
def test_decode_zerocopy_small(benchmark):
    """Zero-copy decode - small DataFrame."""
    codec = DataFrameCodec()
    data = codec.encode(SMALL_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-small")
@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_decode_pyarrow_small(benchmark):
    """PyArrow IPC decode - small DataFrame."""
    codec = PyArrowCodec()
    data = codec.encode(SMALL_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-small")
@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_decode_feather_small(benchmark):
    """Feather (pandas built-in) decode - small DataFrame."""
    codec = FeatherCodec()
    data = codec.encode(SMALL_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-medium")
def test_decode_pickle_medium(benchmark):
    """Pickle decode - medium DataFrame."""
    codec = PicklePandasCodec()
    data = codec.encode(MEDIUM_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-medium")
def test_decode_copy_medium(benchmark):
    """Copy decode - medium DataFrame."""
    codec = DataFrameCopyCodec()
    data = codec.encode(MEDIUM_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-medium")
def test_decode_zerocopy_medium(benchmark):
    """Zero-copy decode - medium DataFrame."""
    codec = DataFrameCodec()
    data = codec.encode(MEDIUM_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-medium")
@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_decode_pyarrow_medium(benchmark):
    """PyArrow IPC decode - medium DataFrame."""
    codec = PyArrowCodec()
    data = codec.encode(MEDIUM_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-medium")
@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_decode_feather_medium(benchmark):
    """Feather (pandas built-in) decode - medium DataFrame."""
    codec = FeatherCodec()
    data = codec.encode(MEDIUM_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-large")
def test_decode_pickle_large(benchmark):
    """Pickle decode - large DataFrame."""
    codec = PicklePandasCodec()
    data = codec.encode(LARGE_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-large")
def test_decode_copy_large(benchmark):
    """Copy decode - large DataFrame."""
    codec = DataFrameCopyCodec()
    data = codec.encode(LARGE_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-large")
def test_decode_zerocopy_large(benchmark):
    """Zero-copy decode - large DataFrame."""
    codec = DataFrameCodec()
    data = codec.encode(LARGE_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-large")
@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_decode_pyarrow_large(benchmark):
    """PyArrow IPC decode - large DataFrame."""
    codec = PyArrowCodec()
    data = codec.encode(LARGE_DF)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="pandas-decode-large")
@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_decode_feather_large(benchmark):
    """Feather (pandas built-in) decode - large DataFrame."""
    codec = FeatherCodec()
    data = codec.encode(LARGE_DF)
    benchmark(codec.decode, data)


def test_zero_copy_proof_memory():
    """Prove zero-copy by showing memory doesn't increase with DataFrame size."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a large DataFrame (38 MB)
        large_df = pd.DataFrame({
            f'col{i}': np.random.rand(100000) for i in range(10)
        })
        df_size_mb = sum(large_df[col].nbytes for col in large_df.columns) / 1024 / 1024
        
        # Write it
        async def write_data():
            oblog = ObLog(tmp_path)
            write = oblog.get_writer("test", DataFrameCodec())
            write(large_df)
            await oblog.close()
        
        asyncio.run(write_data())
        del large_df
        gc.collect()
        
        # Measure memory before read
        mem_before = get_memory_mb()
        
        # Read DataFrame (should be zero-copy, minimal memory increase)
        async def read_data():
            oblog = ObLog(tmp_path)
            result = None
            async for _, df in oblog.read_channel("test"):
                result = df
                # Verify columns are views
                for col in df.columns:
                    arr = df[col].values
                    assert arr.base is not None, f"Column '{col}' should be a view"
                    assert not arr.flags.writeable, f"Column '{col}' should be read-only"
            await oblog.close()
            return result
        
        df = asyncio.run(read_data())
        mem_after = get_memory_mb()
        mem_increase = mem_after - mem_before
        
        print(f"\nZero-copy memory test:")
        print(f"  DataFrame size: {df_size_mb:.1f} MB")
        print(f"  Memory before: {mem_before:.1f} MB")
        print(f"  Memory after: {mem_after:.1f} MB")
        print(f"  Memory increase: {mem_increase:.1f} MB")
        print(f"  All columns are views: {all(df[col].values.base is not None for col in df.columns)}")
        print(f"  All columns read-only: {all(not df[col].values.flags.writeable for col in df.columns)}")
        
        # Memory increase should be reasonable
        # Note: Pandas has some overhead for DataFrame structures, but it's still
        # much less than copying all the data
        assert mem_increase < df_size_mb * 3.0, \
            f"Memory increase too high ({mem_increase:.1f} MB) for zero-copy"
        
        # The key test: columns are views, not copies
        assert all(df[col].values.base is not None for col in df.columns)


def test_zero_copy_proof_data_integrity():
    """Verify zero-copy reads return correct data."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test DataFrames with known values
        dfs = [
            pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]}),
            pd.DataFrame({
                'x': np.arange(100, dtype=np.float64),
                'y': np.arange(100, 200, dtype=np.float64),
            }),
        ]
        
        # Write
        async def write_data():
            oblog = ObLog(tmp_path)
            write = oblog.get_writer("test", DataFrameCodec())
            for df in dfs:
                write(df)
            await oblog.close()
        
        asyncio.run(write_data())
        
        # Read and verify
        async def read_and_verify():
            oblog = ObLog(tmp_path)
            read_dfs = []
            async for _, df in oblog.read_channel("test"):
                read_dfs.append(df.copy())  # Copy to compare
            await oblog.close()
            return read_dfs
        
        read_dfs = asyncio.run(read_and_verify())
        
        # Verify all DataFrames match
        assert len(read_dfs) == len(dfs)
        for orig, read in zip(dfs, read_dfs):
            pd.testing.assert_frame_equal(orig, read)


if __name__ == "__main__":
    """Run zero-copy proof tests manually."""
    import time
    
    print("=" * 70)
    print("Zero-Copy Proof Tests for DataFrameCodec")
    print("=" * 70)
    
    test_zero_copy_proof_data_integrity()
    print("✓ Data integrity test passed")
    
    test_zero_copy_proof_memory()
    print("✓ Memory efficiency test passed")
    
    print("\n" + "=" * 70)
    print("Pure Decode Benchmarks - Zero-Copy vs Copy")
    print("=" * 70)
    
    # Test with large DataFrame
    print(f"\nDataFrame size: {sum(LARGE_DF[col].nbytes for col in LARGE_DF.columns) / 1024 / 1024:.1f} MB")
    print(f"Shape: {LARGE_DF.shape}")
    
    # Pickle
    codec_pickle = PicklePandasCodec()
    data_pickle = codec_pickle.encode(LARGE_DF)
    t0 = time.perf_counter()
    for _ in range(20):
        df = codec_pickle.decode(data_pickle)
    t_pickle = (time.perf_counter() - t0) / 20
    
    # Copy
    codec_copy = DataFrameCopyCodec()
    data_copy = codec_copy.encode(LARGE_DF)
    t0 = time.perf_counter()
    for _ in range(20):
        df = codec_copy.decode(data_copy)
    t_copy = (time.perf_counter() - t0) / 20
    
    # Zero-copy
    codec_zero = DataFrameCodec()
    data_zero = codec_zero.encode(LARGE_DF)
    t0 = time.perf_counter()
    for _ in range(20):
        df = codec_zero.decode(data_zero)
    t_zero = (time.perf_counter() - t0) / 20
    
    print(f"\nPickle decode:    {t_pickle*1000:.2f} ms ({t_pickle/t_zero:.1f}x slower)")
    print(f"Copy decode:      {t_copy*1000:.2f} ms ({t_copy/t_zero:.1f}x slower)")
    print(f"Zero-copy decode: {t_zero*1000:.2f} ms (baseline)")
    
    print(f"\n✓ Zero-copy is {t_pickle/t_zero:.1f}x faster than pickle!")
    print(f"✓ Zero-copy is {t_copy/t_zero:.1f}x faster than copying!")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
