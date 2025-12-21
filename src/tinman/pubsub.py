"""Simple pub-sub implementation for tinman nodes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .timer import VirtualClock


class _Closed:
    """Sentinel value to signal end of stream."""

    pass


_CLOSED = _Closed()


@dataclass
class _TimestampedItem[T]:
    """Message wrapper that carries its publication timestamp."""
    data: T
    timestamp: int


class Out[T]:
    """Output channel from a node.

    Nodes publish data to outputs. The framework automatically
    closes outputs when nodes complete.
    """

    def __init__(self, clock: "VirtualClock | None" = None) -> None:
        self.subscribers: list[In[T]] = []
        self._clock = clock

    def sub(self, maxsize: int = 10) -> "In[T]":
        """Create a subscriber (used internally by framework).
        
        Args:
            maxsize: Maximum queue size for backpressure (default: 10).
        """
        sub: In[T] = In(maxsize=maxsize, clock=self._clock)
        self.subscribers.append(sub)
        return sub

    async def publish(self, data: T) -> None:
        """Publish data to all subscribers."""
        # In fast-forward mode, stamp each message with the current clock time
        # so consumers can schedule at the correct timestamp
        if self._clock is not None:
            item: T | _TimestampedItem[T] = _TimestampedItem(data, self._clock.now())
        else:
            item = data
        for sub in self.subscribers:
            await sub._queue.put(item)

    async def close(self) -> None:
        """Signal all subscribers that the stream has ended.

        Called by framework, not by nodes.
        """
        for sub in self.subscribers:
            await sub._queue.put(_CLOSED)


class In[T]:
    """Input channel to a node.

    Nodes iterate over inputs to receive data.
    """

    def __init__(self, maxsize: int = 10, clock: "VirtualClock | None" = None) -> None:
        """Initialize input channel with bounded queue.
        
        Args:
            maxsize: Maximum queue size for backpressure (default: 10).
            clock: VirtualClock for fast-forward mode coordination (optional).
        """
        self._queue: asyncio.Queue[T | _TimestampedItem[T] | _Closed] = asyncio.Queue(maxsize)
        self._closed = False
        self._clock = clock

    def __aiter__(self) -> "In[T]":
        return self

    async def __anext__(self) -> T:
        if self._closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, _Closed):
            self._closed = True
            raise StopAsyncIteration
        
        # Handle timestamped items in fast-forward mode
        if isinstance(item, _TimestampedItem):
            # Schedule at the publication timestamp for proper interleaving
            if self._clock is not None:
                await self._clock.schedule(item.timestamp)
            return item.data
        
        return item
