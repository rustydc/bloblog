"""Benchmarks for NumpyArrayCodec performance."""

import asyncio

import numpy as np
import pytest

from tinman import ObLogWriter, ObLogReader
from tinman.codecs import NumpyArrayCodec


@pytest.mark.benchmark(group="numpy-write")
def test_numpy_codec_write_small(benchmark, tmp_path):
    """Benchmark writing small arrays (1KB)."""
    arr = np.random.rand(128).astype(np.float64)  # 1KB

    async def write_arrays():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", NumpyArrayCodec())
            for _ in range(100):
                write(arr)

    benchmark(lambda: asyncio.run(write_arrays()))


@pytest.mark.benchmark(group="numpy-write")
def test_numpy_codec_write_large(benchmark, tmp_path):
    """Benchmark writing large arrays (1MB)."""
    arr = np.random.rand(128, 1024).astype(np.float64)  # 1MB

    async def write_arrays():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", NumpyArrayCodec())
            for _ in range(10):
                write(arr)

    benchmark(lambda: asyncio.run(write_arrays()))


@pytest.mark.benchmark(group="numpy-read")
def test_numpy_codec_read_small(benchmark, tmp_path):
    """Benchmark reading small arrays (1KB)."""
    arr = np.random.rand(128).astype(np.float64)

    async def setup():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", NumpyArrayCodec())
            for _ in range(100):
                write(arr)

    asyncio.run(setup())

    async def read_arrays():
        reader = ObLogReader(tmp_path)
        count = 0
        async for _, _ in reader.read_channel("test"):
            count += 1
        return count

    benchmark(lambda: asyncio.run(read_arrays()))


@pytest.mark.benchmark(group="numpy-read")
def test_numpy_codec_read_large(benchmark, tmp_path):
    """Benchmark reading large arrays (1MB) - zero-copy."""
    arr = np.random.rand(128, 1024).astype(np.float64)

    async def setup():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", NumpyArrayCodec())
            for _ in range(10):
                write(arr)

    asyncio.run(setup())

    async def read_arrays():
        reader = ObLogReader(tmp_path)
        arrays = []
        async for _, a in reader.read_channel("test"):
            arrays.append(a)
        return len(arrays)

    benchmark(lambda: asyncio.run(read_arrays()))
