from __future__ import annotations

import asyncio
import mmap
import os
import weakref
from typing import BinaryIO
from time import time_ns
from pathlib import Path
from typing import Callable, AsyncGenerator, Awaitable, TypeVar, Union
from collections import defaultdict
from heapq import heappush, heappop
import struct

T = TypeVar('T')

HEADER_STRUCT = struct.Struct('<QQ')

# Get IOV_MAX from system, fallback to POSIX minimum (16) if unavailable
try:
    IOV_MAX = os.sysconf('SC_IOV_MAX')
except (AttributeError, ValueError, OSError):
    IOV_MAX = 16  # POSIX minimum guaranteed

class BlobLogWriter:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queues: dict[Path, asyncio.Queue[tuple[int, bytes] | None]] = {}
        self.tasks: dict[Path, asyncio.Task] = {}

    def get_writer(self, channel: str) -> Callable[[bytes], None]:
        path = self.log_dir / f"{channel}.bloblog"
        
        if path not in self.queues:
            queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue()
            self.queues[path] = queue
            self.tasks[path] = asyncio.create_task(self._writer_task(path, queue))
        
        queue = self.queues[path]
        
        def write(data: bytes) -> None:
            queue.put_nowait((time_ns(), data))
        
        return write

    async def _writer_task(self, path: Path, queue: asyncio.Queue[tuple[int, bytes] | None]) -> None:
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
                    chunks.append(HEADER_STRUCT.pack(time, len(data)))
                    chunks.append(data)
                
                # writev has a limit (IOV_MAX), so batch if needed
                for i in range(0, len(chunks), IOV_MAX):
                    await asyncio.to_thread(os.writev, fd, chunks[i:i + IOV_MAX])
                
                if item is None:
                    break
        finally:
            await asyncio.to_thread(os.close, fd)

    async def close(self) -> None:
        for queue in self.queues.values():
            queue.put_nowait(None)
        for task in self.tasks.values():
            await task
        self.queues.clear()
        self.tasks.clear()

class BlobLogReader:
    def __init__(self):
        self.handlers: dict[Path, list[Callable[[int, memoryview], Awaitable[None]]]] = defaultdict(list)

    def handle(self, channel: str | Path, callback: Callable[[int, memoryview], Awaitable[None]]) -> None:
        self.handlers[Path(channel)].append(callback)

    async def process(self, speed: float = 0) -> None:
        """Process all registered channels.
        
        Args:
            speed: Playback speed multiplier. 0 for as-fast-as-possible,
                   1.0 for realtime, 2.0 for double speed, 0.5 for half speed, etc.
        """
        paths = list(self.handlers.keys())
        readers = [read_channel(path) for path in paths]
        
        first_log_time: int | None = None
        start_wall_time: int | None = None
        
        async for idx, time, data in amerge(*readers):
            if speed:
                if first_log_time is None:
                    first_log_time = time
                    start_wall_time = time_ns()
                else:
                    # Calculate how long to wait, adjusted for speed
                    assert start_wall_time is not None
                    log_delta = time - first_log_time
                    wall_delta = time_ns() - start_wall_time
                    wait_ns = (log_delta / speed) - wall_delta
                    if wait_ns > 0:
                        await asyncio.sleep(wait_ns / 1_000_000_000)
            
            # Run callbacks for the channel that produced this data
            for callback in self.handlers[paths[idx]]:
                await callback(time, data)


def _close_mmap(mv: memoryview, mm: mmap.mmap, f: BinaryIO) -> None:
    """Weak reference callback to close mmap resources."""
    mv.release()
    mm.close()
    f.close()


async def read_channel(path: Path) -> AsyncGenerator[tuple[int, memoryview], None]:
    """Read a bloblog file, yielding (timestamp, data) tuples.
    
    The memoryview slices remain valid as long as they are referenced.
    Resources are cleaned up via finalizer when no longer needed.
    """
    f = await asyncio.to_thread(open, path, 'rb')
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
            raise ValueError(f"Truncated header in {path}: expected 16 bytes, got {size - offset}")
        time, data_len = HEADER_STRUCT.unpack_from(mm, offset)
        offset += 16
        if size - offset < data_len:
            raise ValueError(f"Truncated data in {path}: expected {data_len} bytes, got {size - offset}")
        data = mv[offset:offset + data_len]  # zero-copy memoryview slice
        offset += data_len
        yield time, data
        count += 1
        if count % batch_size == 0:
            await asyncio.sleep(0)  # Yield to event loop


async def amerge(*iterables: AsyncGenerator[tuple[int, T], None]) -> AsyncGenerator[tuple[int, int, T], None]:
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