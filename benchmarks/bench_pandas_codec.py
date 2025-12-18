"""Benchmarks for pandas DataFrame codec performance."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from tinman import ObLogWriter, ObLogReader
from tinman.codecs import DataFrameCodec


# Test DataFrames
SMALL_DF = pd.DataFrame({
    "a": np.random.rand(100),
    "b": np.random.rand(100),
    "c": np.random.randint(0, 100, 100),
})

LARGE_DF = pd.DataFrame({
    "col1": np.random.rand(100000),
    "col2": np.random.rand(100000),
    "col3": np.random.rand(100000),
    "col4": np.random.randint(0, 10000, 100000),
    "col5": np.random.rand(100000),
})


@pytest.mark.benchmark(group="pandas-write")
def test_pandas_codec_write_small(benchmark, tmp_path):
    """Benchmark writing small DataFrames."""

    async def write_dfs():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", DataFrameCodec())
            for _ in range(100):
                write(SMALL_DF)

    benchmark(lambda: asyncio.run(write_dfs()))


@pytest.mark.benchmark(group="pandas-write")
def test_pandas_codec_write_large(benchmark, tmp_path):
    """Benchmark writing large DataFrames."""

    async def write_dfs():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", DataFrameCodec())
            for _ in range(10):
                write(LARGE_DF)

    benchmark(lambda: asyncio.run(write_dfs()))


@pytest.mark.benchmark(group="pandas-read")
def test_pandas_codec_read_small(benchmark, tmp_path):
    """Benchmark reading small DataFrames."""

    async def setup():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", DataFrameCodec())
            for _ in range(100):
                write(SMALL_DF)

    asyncio.run(setup())

    async def read_dfs():
        reader = ObLogReader(tmp_path)
        count = 0
        async for _, _ in reader.read_channel("test"):
            count += 1
        return count

    benchmark(lambda: asyncio.run(read_dfs()))


@pytest.mark.benchmark(group="pandas-read")
def test_pandas_codec_read_large(benchmark, tmp_path):
    """Benchmark reading large DataFrames - zero-copy."""

    async def setup():
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test", DataFrameCodec())
            for _ in range(10):
                write(LARGE_DF)

    asyncio.run(setup())

    async def read_dfs():
        reader = ObLogReader(tmp_path)
        dfs = []
        async for _, df in reader.read_channel("test"):
            dfs.append(df)
        return len(dfs)

    benchmark(lambda: asyncio.run(read_dfs()))
