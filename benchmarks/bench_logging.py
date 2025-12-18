"""Benchmarks for BlobLog write performance."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinman.bloblog import BlobLogWriter


class TestWriterPerformance:
    """Benchmark BlobLog for various message sizes and patterns."""

    @pytest.fixture
    def tmp_log_dir(self):
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.benchmark(group="write-small")
    def test_write_small_messages(self, benchmark, tmp_log_dir):
        """Benchmark writing many small messages (100 bytes)."""

        async def write_messages():
            async with BlobLogWriter(tmp_log_dir) as writer:
                write = writer.get_writer("bench")

                for _ in range(1000):
                    write(b"x" * 100)

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-medium")
    def test_write_medium_messages(self, benchmark, tmp_log_dir):
        """Benchmark writing medium messages (10KB)."""

        async def write_messages():
            async with BlobLogWriter(tmp_log_dir) as writer:
                write = writer.get_writer("bench")

                for _ in range(1000):
                    write(b"x" * 10_000)

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-large")
    def test_write_large_messages(self, benchmark, tmp_log_dir):
        """Benchmark writing large messages (1MB)."""

        async def write_messages():
            async with BlobLogWriter(tmp_log_dir) as writer:
                write = writer.get_writer("bench")

                for _ in range(100):
                    write(b"x" * 1_000_000)

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-burst")
    def test_write_burst(self, benchmark, tmp_log_dir):
        """Benchmark burst writing 10k messages (tests batching behavior)."""

        async def write_burst():
            async with BlobLogWriter(tmp_log_dir) as writer:
                write = writer.get_writer("bench")

                for _ in range(10_000):
                    write(b"burst" * 20)

        benchmark(lambda: asyncio.run(write_burst()))

    @pytest.mark.benchmark(group="write-multi-channel")
    def test_write_multiple_channels(self, benchmark, tmp_log_dir):
        """Benchmark writing to 10 channels concurrently."""

        async def write_multi():
            async with BlobLogWriter(tmp_log_dir) as writer:
                writers = [writer.get_writer(f"channel_{i}") for i in range(10)]

                for _ in range(1000):
                    for w in writers:
                        w(b"data" * 25)

        benchmark(lambda: asyncio.run(write_multi()))
