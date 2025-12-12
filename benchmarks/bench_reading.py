"""Benchmarks for BlobLogReader and read_channel performance."""

import asyncio
from pathlib import Path

import pytest

from tinman.bloblog import BlobLog, amerge


def _read_bloblog_channel(log_file_path: Path):
    """Helper to read a channel from its .blog file path."""
    channel_name = log_file_path.stem
    log_dir = log_file_path.parent
    return BlobLog(log_dir)._read_bloblog_channel(channel_name)


class TestReaderPerformance:
    """Benchmark BlobLogReader for various read patterns."""

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
    def xlarge_log_file(self, tmp_path_factory):
        """Create a log file with 100 extra large messages (10MB each)."""

        async def create_log():
            tmpdir = tmp_path_factory.mktemp("xlarge_logs")
            log_dir = Path(tmpdir)
            writer = BlobLog(log_dir)
            write = writer.get_writer("xlarge")

            for i in range(100):
                write(b"x" * 10_000_000)

            await writer.close()
            return log_dir / "xlarge.bloblog"

        return asyncio.run(create_log())

    @pytest.fixture(scope="class")
    def xxlarge_log_file(self, tmp_path_factory):
        """Create a log file with 10 very large messages (100MB each)."""

        async def create_log():
            tmpdir = tmp_path_factory.mktemp("xxlarge_logs")
            log_dir = Path(tmpdir)
            writer = BlobLog(log_dir)
            write = writer.get_writer("xxlarge")

            for i in range(10):
                write(b"x" * 100_000_000)

            await writer.close()
            return log_dir / "xxlarge.bloblog"

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

    @pytest.mark.benchmark(group="read-large")
    def test_read_large_messages(self, benchmark, large_log_file):
        """Benchmark reading large messages."""

        async def read_all():
            count = 0
            total_bytes = 0
            async for _time, data in _read_bloblog_channel(large_log_file):
                count += 1
                total_bytes += len(data)
            return count, total_bytes

        result = benchmark(lambda: asyncio.run(read_all()))
        assert result[0] == 1_000

    @pytest.mark.benchmark(group="read-memoryview")
    def test_memoryview_slice_overhead(self, benchmark, small_log_file):
        """Benchmark memoryview slicing overhead vs copying."""

        async def read_with_slices():
            """Current implementation - zero-copy slices."""
            data_list = []
            async for _time, data in _read_bloblog_channel(small_log_file):
                data_list.append(data)  # Keep memoryview slice
            # Convert all at once
            return [bytes(d) for d in data_list]

        result = benchmark(lambda: asyncio.run(read_with_slices()))
        assert len(result) == 10_000

    @pytest.mark.benchmark(group="read-copy-comparison")
    def test_immediate_copy_vs_delayed(self, benchmark, small_log_file):
        """Compare immediate byte copying vs delayed memoryview access."""

        async def read_with_immediate_copy():
            """Copy bytes immediately."""
            data_list = []
            async for _time, data in _read_bloblog_channel(small_log_file):
                data_list.append(bytes(data))  # Immediate copy
            return data_list

        result = benchmark(lambda: asyncio.run(read_with_immediate_copy()))
        assert len(result) == 10_000

    @pytest.mark.benchmark(group="read-throughput")
    def test_read_throughput(self, benchmark, large_log_file):
        """Measure throughput in MB/s."""

        async def measure_throughput():
            total_bytes = 0
            async for _time, data in _read_bloblog_channel(large_log_file):
                total_bytes += len(data)
            return total_bytes

        result = benchmark(lambda: asyncio.run(measure_throughput()))
        # 1000 messages * 100KB = ~100MB
        assert result == 1_000 * 100_000

    @pytest.mark.benchmark(group="read-throughput-xlarge")
    def test_read_throughput_xlarge(self, benchmark, xlarge_log_file):
        """Measure throughput with 10MB messages."""

        async def measure_throughput():
            total_bytes = 0
            async for _time, data in _read_bloblog_channel(xlarge_log_file):
                total_bytes += len(data)
            return total_bytes

        result = benchmark(lambda: asyncio.run(measure_throughput()))
        # 100 messages * 10MB = ~1GB
        assert result == 100 * 10_000_000

    @pytest.mark.benchmark(group="read-throughput-xxlarge")
    def test_read_throughput_xxlarge(self, benchmark, xxlarge_log_file):
        """Measure throughput with 100MB messages."""

        async def measure_throughput():
            total_bytes = 0
            async for _time, data in _read_bloblog_channel(xxlarge_log_file):
                total_bytes += len(data)
            return total_bytes

        result = benchmark(lambda: asyncio.run(measure_throughput()))
        # 10 messages * 100MB = ~1GB
        assert result == 10 * 100_000_000

    @pytest.mark.benchmark(group="reader-callbacks")
    def test_reader_with_callbacks(self, benchmark, small_log_file):
        """Benchmark read_channel directly."""

        async def read_with_channel():
            results = []
            async for _time, data in _read_bloblog_channel(small_log_file):
                results.append(bytes(data))
            return results

        result = benchmark(lambda: asyncio.run(read_with_channel()))
        assert len(result) == 10_000

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

    @pytest.mark.benchmark(group="batch-size-test")
    def test_batch_size_impact(self, benchmark, small_log_file):
        """Test impact of batch_size=1000 for yielding to event loop."""

        async def read_all_batched():
            count = 0
            async for _time, _data in _read_bloblog_channel(small_log_file):
                count += 1
                # Current implementation yields every 1000 items
            return count

        result = benchmark(lambda: asyncio.run(read_all_batched()))
        assert result == 10_000
