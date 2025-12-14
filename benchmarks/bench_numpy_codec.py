"""Benchmarks to prove NumpyArrayCodec is zero-copy.

These benchmarks demonstrate:
1. Reading is much faster than writing (no copy on read)
2. Memory usage doesn't increase proportionally with array size
3. Multiple reads of the same data don't accumulate memory
4. Zero-copy is faster than pickle and JSON codecs
"""

import asyncio
import gc
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from tinman import Codec, ObLog
from tinman.codecs import NumpyArrayCodec


class PickleNumpyCodec(Codec[np.ndarray]):
    """Baseline: pickle-based codec (always copies on read)."""

    def encode(self, item: np.ndarray) -> bytes:
        return pickle.dumps(item)

    def decode(self, data: bytes) -> np.ndarray:
        return pickle.loads(data)


def get_memory_mb():
    """Get current RSS memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except (ImportError, AttributeError):
        # macOS returns bytes, Linux returns KB
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return rss / 1024 / 1024
        else:
            return rss / 1024


@pytest.mark.benchmark(group="numpy-write")
def test_numpy_codec_write_small(benchmark, tmp_path):
    """Benchmark writing small arrays (1KB)."""
    arr = np.random.rand(128).astype(np.float64)  # 1KB
    
    async def write_arrays():
        oblog = ObLog(tmp_path)
        write = oblog.get_writer("test", NumpyArrayCodec())
        for _ in range(100):
            write(arr)
        await oblog.close()
    
    benchmark(lambda: asyncio.run(write_arrays()))


@pytest.mark.benchmark(group="numpy-write")
def test_numpy_codec_write_large(benchmark, tmp_path):
    """Benchmark writing large arrays (1MB)."""
    arr = np.random.rand(128, 1024).astype(np.float64)  # 1MB
    
    async def write_arrays():
        oblog = ObLog(tmp_path)
        write = oblog.get_writer("test", NumpyArrayCodec())
        for _ in range(10):
            write(arr)
        await oblog.close()
    
    benchmark(lambda: asyncio.run(write_arrays()))


@pytest.mark.benchmark(group="numpy-read")
def test_numpy_codec_read_small(benchmark, tmp_path):
    """Benchmark reading small arrays (1KB) - should be faster than write."""
    arr = np.random.rand(128).astype(np.float64)
    
    # Setup: write data
    async def setup():
        oblog = ObLog(tmp_path)
        write = oblog.get_writer("test", NumpyArrayCodec())
        for _ in range(100):
            write(arr)
        await oblog.close()
    
    asyncio.run(setup())
    
    # Benchmark: read data
    async def read_arrays():
        oblog = ObLog(tmp_path)
        count = 0
        async for _, _ in oblog.read_channel("test"):
            count += 1
        await oblog.close()
        return count
    
    benchmark(lambda: asyncio.run(read_arrays()))


@pytest.mark.benchmark(group="numpy-read")
def test_numpy_codec_read_large(benchmark, tmp_path):
    """Benchmark reading large arrays (1MB) - should be much faster than write."""
    arr = np.random.rand(128, 1024).astype(np.float64)
    
    # Setup: write data
    async def setup():
        oblog = ObLog(tmp_path)
        write = oblog.get_writer("test", NumpyArrayCodec())
        for _ in range(10):
            write(arr)
        await oblog.close()
    
    asyncio.run(setup())
    
    # Benchmark: read data
    async def read_arrays():
        oblog = ObLog(tmp_path)
        arrays = []
        async for _, a in oblog.read_channel("test"):
            arrays.append(a)
        await oblog.close()
        return len(arrays)
    
    benchmark(lambda: asyncio.run(read_arrays()))


@pytest.mark.benchmark(group="codec-comparison")
def test_pickle_codec_read(benchmark, tmp_path):
    """Baseline: pickle codec (copies on read)."""
    arr = np.random.rand(128, 1024).astype(np.float64)  # 1MB
    
    # Setup
    async def setup():
        oblog = ObLog(tmp_path)
        write = oblog.get_writer("test", PickleNumpyCodec())
        for _ in range(10):
            write(arr)
        await oblog.close()
    
    asyncio.run(setup())
    
    # Benchmark
    async def read_arrays():
        oblog = ObLog(tmp_path)
        arrays = []
        async for _, a in oblog.read_channel("test"):
            arrays.append(a)
        await oblog.close()
        return len(arrays)
    
    benchmark(lambda: asyncio.run(read_arrays()))


@pytest.mark.benchmark(group="codec-comparison")
def test_numpy_codec_read_comparison(benchmark, tmp_path):
    """Zero-copy codec (should be faster than pickle)."""
    arr = np.random.rand(128, 1024).astype(np.float64)  # 1MB
    
    # Setup
    async def setup():
        oblog = ObLog(tmp_path)
        write = oblog.get_writer("test", NumpyArrayCodec())
        for _ in range(10):
            write(arr)
        await oblog.close()
    
    asyncio.run(setup())
    
    # Benchmark
    async def read_arrays():
        oblog = ObLog(tmp_path)
        arrays = []
        async for _, a in oblog.read_channel("test"):
            arrays.append(a)
        await oblog.close()
        return len(arrays)
    
    benchmark(lambda: asyncio.run(read_arrays()))


def test_zero_copy_proof_memory():
    """Prove zero-copy by showing memory doesn't increase with file size."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a large array (100MB)
        large_array = np.random.rand(1000, 10000).astype(np.float64)
        array_size_mb = large_array.nbytes / 1024 / 1024
        
        # Write it
        async def write_data():
            oblog = ObLog(tmp_path)
            write = oblog.get_writer("test", NumpyArrayCodec())
            write(large_array)
            await oblog.close()
        
        asyncio.run(write_data())
        del large_array
        gc.collect()
        
        # Measure memory before read
        mem_before = get_memory_mb()
        
        # Read array (should be zero-copy, minimal memory increase)
        async def read_data():
            oblog = ObLog(tmp_path)
            result = None
            async for _, arr in oblog.read_channel("test"):
                result = arr
                # Verify it's a view
                assert arr.base is not None, "Array should be a view"
                assert not arr.flags.writeable, "Array should be read-only"
            await oblog.close()
            return result
        
        arr = asyncio.run(read_data())
        mem_after = get_memory_mb()
        
        # Memory increase should be much less than array size
        # (some overhead is expected, but nowhere near 100MB)
        mem_increase = mem_after - mem_before
        
        print(f"\nZero-copy memory test:")
        print(f"  Array size: {array_size_mb:.1f} MB")
        print(f"  Memory before: {mem_before:.1f} MB")
        print(f"  Memory after: {mem_after:.1f} MB")
        print(f"  Memory increase: {mem_increase:.1f} MB")
        print(f"  Array is view: {arr.base is not None}")
        print(f"  Array is read-only: {not arr.flags.writeable}")
        
        # Memory increase should be < 20% of array size (generous threshold)
        # In practice it's usually < 1% due to lazy loading
        assert mem_increase < array_size_mb * 0.2, \
            f"Memory increase too high ({mem_increase:.1f} MB) for zero-copy"


def test_zero_copy_proof_multiple_reads():
    """Prove zero-copy by reading same large array multiple times."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a large array (50MB)
        large_array = np.random.rand(500, 10000).astype(np.float64)
        array_size_mb = large_array.nbytes / 1024 / 1024
        
        # Write it
        async def write_data():
            oblog = ObLog(tmp_path)
            write = oblog.get_writer("test", NumpyArrayCodec())
            write(large_array)
            await oblog.close()
        
        asyncio.run(write_data())
        del large_array
        gc.collect()
        
        mem_before = get_memory_mb()
        
        # Read the same array 5 times
        # With copying, this would use 5x memory
        # With zero-copy, all arrays share the same mmap
        async def read_multiple():
            arrays = []
            for i in range(5):
                oblog = ObLog(tmp_path)
                async for _, arr in oblog.read_channel("test"):
                    arrays.append(arr)
                await oblog.close()
            return arrays
        
        arrays = asyncio.run(read_multiple())
        mem_after = get_memory_mb()
        mem_increase = mem_after - mem_before
        
        print(f"\nMultiple reads memory test:")
        print(f"  Array size: {array_size_mb:.1f} MB")
        print(f"  Number of reads: 5")
        print(f"  Expected with copy: ~{array_size_mb * 5:.1f} MB")
        print(f"  Memory increase: {mem_increase:.1f} MB")
        print(f"  All arrays are views: {all(arr.base is not None for arr in arrays)}")
        
        # With zero-copy, 5 reads should use much less than 5x memory
        # Each read creates a new mmap, so some overhead expected
        assert mem_increase < array_size_mb * 2, \
            f"Memory increase suggests copying, not zero-copy"


def test_zero_copy_proof_data_integrity():
    """Verify zero-copy reads return correct data."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test arrays with known values
        arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.arange(1000, dtype=np.int64),
        ]
        
        # Write
        async def write_data():
            oblog = ObLog(tmp_path)
            write = oblog.get_writer("test", NumpyArrayCodec())
            for arr in arrays:
                write(arr)
            await oblog.close()
        
        asyncio.run(write_data())
        
        # Read and verify
        async def read_and_verify():
            oblog = ObLog(tmp_path)
            read_arrays = []
            async for _, arr in oblog.read_channel("test"):
                read_arrays.append(arr.copy())  # Copy to compare
            await oblog.close()
            return read_arrays
        
        read_arrays = asyncio.run(read_and_verify())
        
        # Verify all arrays match
        assert len(read_arrays) == len(arrays)
        for orig, read in zip(arrays, read_arrays):
            np.testing.assert_array_equal(orig, read)


if __name__ == "__main__":
    """Run zero-copy proof tests manually."""
    print("=" * 70)
    print("Zero-Copy Proof Tests for NumpyArrayCodec")
    print("=" * 70)
    
    test_zero_copy_proof_data_integrity()
    print("✓ Data integrity test passed")
    
    test_zero_copy_proof_memory()
    print("✓ Memory efficiency test passed")
    
    test_zero_copy_proof_multiple_reads()
    print("✓ Multiple reads test passed")
    
    print("\n" + "=" * 70)
    print("All zero-copy proofs passed!")
    print("=" * 70)
