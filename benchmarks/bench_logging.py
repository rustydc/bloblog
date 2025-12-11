"""Benchmarks for BlobLogWriter write performance."""

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bloblog.bloblog import BlobLogWriter


class TestWriterPerformance:
    """Benchmark BlobLogWriter for various message sizes and patterns."""

    @pytest.fixture
    def tmp_log_dir(self):
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.benchmark(group="write-small")
    def test_write_small_messages_sync(self, benchmark, tmp_log_dir):
        """Benchmark writing many small messages (100 bytes)."""

        async def write_messages():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("bench")

            for _ in range(1000):
                write(b"x" * 100)

            await writer.close()

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-medium")
    def test_write_medium_messages_sync(self, benchmark, tmp_log_dir):
        """Benchmark writing medium messages (10KB)."""

        async def write_messages():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("bench")

            for _ in range(1000):
                write(b"x" * 10_000)

            await writer.close()

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-large")
    def test_write_large_messages_sync(self, benchmark, tmp_log_dir):
        """Benchmark writing large messages (1MB)."""

        async def write_messages():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("bench")

            for _ in range(100):
                write(b"x" * 1_000_000)

            await writer.close()

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-xlarge")
    def test_write_xlarge_messages_sync(self, benchmark, tmp_log_dir):
        """Benchmark writing extra large messages (10MB)."""

        async def write_messages():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("bench")

            for _ in range(10):
                write(b"x" * 10_000_000)

            await writer.close()

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-xxlarge")
    def test_write_xxlarge_messages_sync(self, benchmark, tmp_log_dir):
        """Benchmark writing very large messages (100MB)."""

        async def write_messages():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("bench")

            # Just 2 messages to keep benchmark time reasonable
            for _ in range(2):
                write(b"x" * 100_000_000)

            await writer.close()

        benchmark(lambda: asyncio.run(write_messages()))

    @pytest.mark.benchmark(group="write-burst")
    def test_write_burst(self, benchmark, tmp_log_dir):
        """Benchmark burst writing (tests batching behavior)."""

        async def write_burst():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("bench")

            # Write 10000 messages as fast as possible
            for _ in range(10_000):
                write(b"burst" * 20)

            await writer.close()

        benchmark(lambda: asyncio.run(write_burst()))

    @pytest.mark.benchmark(group="write-multi-channel")
    def test_write_multiple_channels(self, benchmark, tmp_log_dir):
        """Benchmark writing to multiple channels concurrently."""

        async def write_multi():
            writer = BlobLogWriter(tmp_log_dir)
            writers = [writer.get_writer(f"channel_{i}") for i in range(10)]

            for _ in range(1000):
                for w in writers:
                    w(b"data" * 25)

            await writer.close()

        benchmark(lambda: asyncio.run(write_multi()))

    @pytest.mark.benchmark(group="writev-comparison")
    def test_writev_vs_single_writes(self, benchmark, tmp_log_dir):
        """Compare writev batching vs individual writes."""

        async def write_with_writev():
            """Current implementation using writev."""
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("writev_bench")

            # Burst writes to trigger batching
            for _ in range(5000):
                write(b"x" * 100)

            await writer.close()

        benchmark(lambda: asyncio.run(write_with_writev()))

    @pytest.mark.benchmark(group="iov-max-test")
    def test_iov_max_batching(self, benchmark, tmp_log_dir):
        """Test performance with IOV_MAX boundaries."""
        # Get actual IOV_MAX value
        try:
            iov_max = os.sysconf("SC_IOV_MAX")
        except (AttributeError, ValueError, OSError):
            iov_max = 16

        async def write_iov_max():
            writer = BlobLogWriter(tmp_log_dir)
            write = writer.get_writer("iov_bench")

            # Write exactly IOV_MAX * 2 messages to test batching logic
            for _ in range(iov_max * 2):
                write(b"test" * 25)

            await writer.close()

        benchmark(lambda: asyncio.run(write_iov_max()))
