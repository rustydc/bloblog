"""Simple pub-sub implementation for bloblog nodes."""

import asyncio


class _Closed:
    """Sentinel value to signal end of stream."""

    pass


_CLOSED = _Closed()


class Out[T]:
    """Output channel from a node.

    Nodes publish data to outputs. The framework automatically
    closes outputs when nodes complete.
    """

    def __init__(self) -> None:
        self.subscribers: list[In[T]] = []

    def sub(self) -> "In[T]":
        """Create a subscriber (used internally by framework)."""
        sub: In[T] = In()
        self.subscribers.append(sub)
        return sub

    async def publish(self, data: T) -> None:
        """Publish data to all subscribers."""
        for sub in self.subscribers:
            await sub._queue.put(data)

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

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | _Closed] = asyncio.Queue(10)
        self._closed = False

    def __aiter__(self) -> "In[T]":
        return self

    async def __anext__(self) -> T:
        if self._closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, _Closed):
            self._closed = True
            raise StopAsyncIteration
        return item
