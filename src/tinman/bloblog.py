"""Tinman binary log format and I/O implementation."""

from __future__ import annotations

import asyncio
import mmap
import os
import struct
import weakref
from collections.abc import AsyncGenerator, Buffer, Callable
from heapq import heappop, heappush
from pathlib import Path
from time import time_ns
from typing import BinaryIO

HEADER_STRUCT = struct.Struct("<QQ")

# Get IOV_MAX from system, fallback to POSIX minimum (16) if unavailable
try:
    IOV_MAX = os.sysconf("SC_IOV_MAX")
except (AttributeError, ValueError, OSError):
    IOV_MAX = 16  # POSIX minimum guaranteed


class BlobLog:
    """Binary log directory containing multiple channels.

    Each channel is a separate .blog file storing (timestamp, raw bytes) records.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._queues: dict[str, asyncio.Queue[tuple[int, Buffer] | None]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def get_writer(self, channel: str) -> Callable[[Buffer], None]:
        """Get a write function for a channel.

        Returns the same writer if called multiple times for the same channel.
        """
        if channel not in self._queues:
            queue: asyncio.Queue[tuple[int, Buffer] | None] = asyncio.Queue()
            self._queues[channel] = queue
            path = self.log_dir / f"{channel}.blog"
            self._tasks[channel] = asyncio.create_task(self._writer_task(path, queue))

        queue = self._queues[channel]

        def write(data: Buffer) -> None:
            queue.put_nowait((time_ns(), data))

        return write

    async def _writer_task(
        self, path: Path, queue: asyncio.Queue[tuple[int, Buffer] | None]
    ) -> None:
        """Background task that writes queued data to disk."""
        fd = await asyncio.to_thread(os.open, path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break

                # Batch all pending items
                items = [item]
                while not queue.empty():
                    item = queue.get_nowait()
                    if item is None:
                        break
                    items.append(item)

                # Write all at once, respecting IOV_MAX limit
                chunks = []
                for time, data in items:
                    data_mv = memoryview(data)
                    chunks.append(HEADER_STRUCT.pack(time, len(data_mv)))
                    chunks.append(data_mv)

                # writev has a limit (IOV_MAX), so batch if needed
                for i in range(0, len(chunks), IOV_MAX):
                    await asyncio.to_thread(os.writev, fd, chunks[i : i + IOV_MAX])

                if item is None:
                    break
        finally:
            await asyncio.to_thread(os.close, fd)

    async def close(self) -> None:
        """Close all channels and flush pending writes."""
        for queue in self._queues.values():
            queue.put_nowait(None)
        for task in self._tasks.values():
            await task
        self._queues.clear()
        self._tasks.clear()

    async def read_channel(self, channel: str) -> AsyncGenerator[tuple[int, memoryview], None]:
        """Read all records from a channel, yielding (timestamp, data) tuples.

        The memoryview slices remain valid as long as they are referenced.
        Resources are cleaned up via finalizer when no longer needed.
        """
        path = self.log_dir / f"{channel}.blog"
        f = await asyncio.to_thread(open, path, "rb")
        mm = await asyncio.to_thread(mmap.mmap, f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)

        # Set up finalizer to clean up when memoryview is garbage collected
        weakref.finalize(mv, _close_mmap, mv, mm, f)

        offset = 0
        size = len(mm)
        batch_size = 1000  # Yield to event loop periodically
        count = 0

        while offset < size:
            if size - offset < 16:
                raise ValueError(
                    f"Truncated header in {path}: expected 16 bytes, got {size - offset}"
                )
            time, data_len = HEADER_STRUCT.unpack_from(mm, offset)
            offset += 16
            if size - offset < data_len:
                raise ValueError(
                    f"Truncated data in {path}: expected {data_len} bytes, got {size - offset}"
                )
            data = mv[offset : offset + data_len]  # zero-copy memoryview slice
            offset += data_len
            yield time, data
            count += 1
            if count % batch_size == 0:
                await asyncio.sleep(0)  # Yield to event loop


def _close_mmap(mv: memoryview, mm: mmap.mmap, f: BinaryIO) -> None:
    """Weak reference callback to close mmap resources."""
    mv.release()
    mm.close()
    f.close()


async def amerge[T](
    *iterables: AsyncGenerator[tuple[int, T], None],
) -> AsyncGenerator[tuple[int, int, T], None]:
    """Merge async iterables in sorted order by timestamp.

    Yields (source_index, timestamp, item) tuples.

    Note: Items are held briefly until yielded. Callers should consume each item
    before the next iteration to avoid holding references.
    """
    # Each heap entry: (time, source_index, item, iterator)
    # We must store item because we've already consumed it from the iterator
    heap: list[tuple[int, int, T, AsyncGenerator[tuple[int, T], None]]] = []

    for i, it in enumerate(iterables):
        try:
            time, item = await anext(it)
            heappush(heap, (time, i, item, it))
        except StopAsyncIteration:
            pass

    while heap:
        time, i, item, it = heappop(heap)
        yield i, time, item
        # Clear reference to item before fetching next
        del item
        try:
            time, next_item = await anext(it)
            heappush(heap, (time, i, next_item, it))
        except StopAsyncIteration:
            pass
