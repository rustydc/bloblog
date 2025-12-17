"""Benchmarks for BlobLog read performance."""

import asyncio
from pathlib import Path

import pytest

from tinman.bloblog import BlobLog, amerge


def _read_bloblog_channel(log_file_path: Path):
    """Helper to read a channel from its .blog file path."""
    channel_name = log_file_path.stem
    log_dir = log_file_path.parent
    return BlobLog(log_dir).read_channel(channel_name)


class TestReaderPerformance:
    """Benchmark BlobLog read operations."""

    @pytest.fixture(scope="class")
    def small_log_file(self, tmp_path_factory):
        """Create a log file with 10,000 small messages."""

        async def create_log():
            tmpdir = tmp_path_factory.mktemp("small_logs")
            log_dir = Path(tmpdir)
            writer = BlobLog(log_dir)
            write = writer.get_writer("small")

            for i in range(10_000):
                write(f"message_{i}".encode())

            await writer.close()
            return log_dir / "small.bloblog"

        return asyncio.run(create_log())

    @pytest.fixture(scope="class")
    def large_log_file(self, tmp_path_factory):
        """Create a log file with 1,000 large messages (100KB each)."""

        async def create_log():
            tmpdir = tmp_path_factory.mktemp("large_logs")
            log_dir = Path(tmpdir)
            writer = BlobLog(log_dir)
            write = writer.get_writer("large")

            for i in range(1_000):
                write(b"x" * 100_000)

            await writer.close()
            return log_dir / "large.bloblog"

        return asyncio.run(create_log())

    @pytest.fixture(scope="class")
    def multi_channel_logs(self, tmp_path_factory):
        """Create multiple log files for merge testing."""

        async def create_logs():
            tmpdir = tmp_path_factory.mktemp("multi_logs")
            log_dir = Path(tmpdir)
            writer = BlobLog(log_dir)

            # Create 5 channels with 1000 messages each
            writers = [writer.get_writer(f"channel_{i}") for i in range(5)]
            for _ in range(1000):
                for w in writers:
                    w(b"data" * 25)

            await writer.close()
            return log_dir

        return asyncio.run(create_logs())

    @pytest.mark.benchmark(group="read-small")
    def test_read_small_messages(self, benchmark, small_log_file):
        """Benchmark reading many small messages."""

        async def read_all():
            count = 0
            async for _time, _data in _read_bloblog_channel(small_log_file):
                count += 1
            return count

        result = benchmark(lambda: asyncio.run(read_all()))
        assert result == 10_000

    @pytest.mark.benchmark(group="read-throughput")
    def test_read_throughput(self, benchmark, large_log_file):
        """Measure read throughput with large messages (~100MB total)."""

        async def measure_throughput():
            total_bytes = 0
            async for _time, data in _read_bloblog_channel(large_log_file):
                total_bytes += len(data)
            return total_bytes

        result = benchmark(lambda: asyncio.run(measure_throughput()))
        # 1000 messages * 100KB = ~100MB
        assert result == 1_000 * 100_000

    @pytest.mark.benchmark(group="amerge")
    def test_amerge_performance(self, benchmark, multi_channel_logs):
        """Benchmark amerge with multiple channels."""

        async def merge_channels():
            count = 0
            readers = [
                _read_bloblog_channel(multi_channel_logs / f"channel_{i}.bloblog") for i in range(5)
            ]

            async for _idx, _time, _data in amerge(*readers):
                count += 1

            return count

        result = benchmark(lambda: asyncio.run(merge_channels()))
        # 5 channels * 1000 messages = 5000
        assert result == 5_000
