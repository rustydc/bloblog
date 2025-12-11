import asyncio
from pathlib import Path

import pytest

from bloblog.bloblog import HEADER_STRUCT, BlobLogWriter, amerge, read_channel


class TestBlobLogWriter:
    @pytest.mark.asyncio
    async def test_write_single_blob(self, tmp_path: Path):
        writer = BlobLogWriter(tmp_path)
        write = writer.get_writer("test")
        write(b"hello")
        await writer.close()

        # Verify file contents
        log_file = tmp_path / "test.bloblog"
        data = log_file.read_bytes()
        assert len(data) == 16 + 5  # header + "hello"
        time, length = HEADER_STRUCT.unpack_from(data, 0)
        assert length == 5
        assert data[16:] == b"hello"
        assert time > 0

    @pytest.mark.asyncio
    async def test_write_multiple_blobs(self, tmp_path: Path):
        writer = BlobLogWriter(tmp_path)
        write = writer.get_writer("test")
        write(b"one")
        write(b"two")
        write(b"three")
        await writer.close()

        # Verify file contents
        log_file = tmp_path / "test.bloblog"
        data = log_file.read_bytes()
        offset = 0
        expected = [b"one", b"two", b"three"]
        for expected_data in expected:
            time, length = HEADER_STRUCT.unpack_from(data, offset)
            offset += 16
            assert length == len(expected_data)
            assert data[offset : offset + length] == expected_data
            offset += length
        assert offset == len(data)

    @pytest.mark.asyncio
    async def test_write_to_multiple_channels(self, tmp_path: Path):
        writer = BlobLogWriter(tmp_path)
        write1 = writer.get_writer("channel1")
        write2 = writer.get_writer("channel2")

        write1(b"channel1-data")
        write2(b"channel2-data")
        await writer.close()

        # Verify both files exist and have correct content
        log_file1 = tmp_path / "channel1.bloblog"
        log_file2 = tmp_path / "channel2.bloblog"
        assert log_file1.exists()
        assert log_file2.exists()

        data1 = log_file1.read_bytes()
        _, length1 = HEADER_STRUCT.unpack_from(data1, 0)
        assert data1[16 : 16 + length1] == b"channel1-data"

        data2 = log_file2.read_bytes()
        _, length2 = HEADER_STRUCT.unpack_from(data2, 0)
        assert data2[16 : 16 + length2] == b"channel2-data"

    @pytest.mark.asyncio
    async def test_get_writer_returns_same_queue_for_same_channel(self, tmp_path: Path):
        writer = BlobLogWriter(tmp_path)
        _ = writer.get_writer("test")
        _ = writer.get_writer("test")

        # Should use the same queue
        assert len(writer.queues) == 1
        await writer.close()


class TestReadChannel:
    @pytest.mark.asyncio
    async def test_read_single_blob(self, tmp_path: Path):
        log_file = tmp_path / "test.bloblog"

        # Write test data
        writer = BlobLogWriter(tmp_path)
        write = writer.get_writer("test")
        write(b"hello")
        await writer.close()

        # Read it back
        results: list[tuple[int, bytes]] = []
        async for time, data in read_channel(log_file):
            results.append((time, bytes(data)))  # copy memoryview to bytes

        assert len(results) == 1
        assert results[0][1] == b"hello"
        assert results[0][0] > 0

    @pytest.mark.asyncio
    async def test_read_multiple_blobs_in_order(self, tmp_path: Path):
        log_file = tmp_path / "test.bloblog"

        # Write test data with small delays to ensure ordering
        writer = BlobLogWriter(tmp_path)
        write = writer.get_writer("test")
        write(b"first")
        await asyncio.sleep(0.001)
        write(b"second")
        await asyncio.sleep(0.001)
        write(b"third")
        await writer.close()

        # Read it back
        results: list[bytes] = []
        async for _time, data in read_channel(log_file):
            results.append(bytes(data))

        assert results == [b"first", b"second", b"third"]

    @pytest.mark.asyncio
    async def test_read_multiple_channels_merged_by_time(self, tmp_path: Path):
        log_file1 = tmp_path / "channel1.bloblog"
        log_file2 = tmp_path / "channel2.bloblog"

        # Write to channel 1
        writer = BlobLogWriter(tmp_path)
        write1 = writer.get_writer("channel1")
        write1(b"c1-first")
        await asyncio.sleep(0.002)
        write1(b"c1-second")
        await writer.close()

        # Write to channel 2 (with timestamps between channel 1's)
        writer = BlobLogWriter(tmp_path)
        write2 = writer.get_writer("channel2")
        await asyncio.sleep(0.001)
        write2(b"c2-first")
        await asyncio.sleep(0.002)
        write2(b"c2-second")
        await writer.close()

        # Read both channels merged using amerge
        results: list[bytes] = []
        reader1 = read_channel(log_file1)
        reader2 = read_channel(log_file2)
        
        async for _idx, _time, data in amerge(reader1, reader2):
            results.append(bytes(data))

        # All 4 items should be present
        assert len(results) == 4
        assert set(results) == {b"c1-first", b"c1-second", b"c2-first", b"c2-second"}


class TestAmerge:
    @pytest.mark.asyncio
    async def test_merge_empty(self):
        results = [item async for item in amerge()]
        assert results == []

    @pytest.mark.asyncio
    async def test_merge_single_iterable(self):
        async def gen():
            yield 1, "a"
            yield 3, "c"
            yield 5, "e"

        results = [(idx, time, item) async for idx, time, item in amerge(gen())]
        assert results == [(0, 1, "a"), (0, 3, "c"), (0, 5, "e")]

    @pytest.mark.asyncio
    async def test_merge_two_iterables(self):
        async def gen1():
            yield 1, "a"
            yield 3, "c"
            yield 5, "e"

        async def gen2():
            yield 2, "b"
            yield 4, "d"
            yield 6, "f"

        results = [item async for idx, time, item in amerge(gen1(), gen2())]
        assert results == ["a", "b", "c", "d", "e", "f"]

    @pytest.mark.asyncio
    async def test_merge_uneven_iterables(self):
        async def gen1():
            yield 1, "a"

        async def gen2():
            yield 2, "b"
            yield 3, "c"
            yield 4, "d"

        results = [item async for idx, time, item in amerge(gen1(), gen2())]
        assert results == ["a", "b", "c", "d"]


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_large_data(self, tmp_path: Path):
        log_file = tmp_path / "test.bloblog"
        large_blob = b"x" * 1_000_000  # 1MB

        writer = BlobLogWriter(tmp_path)
        write = writer.get_writer("test")
        write(large_blob)
        await writer.close()

        results: list[bytes] = []
        async for _time, data in read_channel(log_file):
            results.append(bytes(data))

        assert len(results) == 1
        assert results[0] == large_blob

    @pytest.mark.asyncio
    async def test_many_small_writes(self, tmp_path: Path):
        log_file = tmp_path / "test.bloblog"
        count = 1000

        writer = BlobLogWriter(tmp_path)
        write = writer.get_writer("test")
        for i in range(count):
            write(f"item-{i}".encode())
        await writer.close()

        results: list[bytes] = []
        async for _time, data in read_channel(log_file):
            results.append(bytes(data))

        assert len(results) == count
        for i, data in enumerate(results):
            assert data == f"item-{i}".encode()
