import asyncio

class _Closed:
    """Sentinel value to signal end of stream."""
    pass

_CLOSED = _Closed()

class Pub[T]:
    def __init__(self) -> None:
        self.subscribers: list[Sub[T]] = []

    def sub(self) -> "Sub[T]":
        sub: Sub[T] = Sub()
        self.subscribers.append(sub)
        return sub
    
    async def publish(self, data: T) -> None:
        async with asyncio.TaskGroup() as tg:
            for sub in self.subscribers:
                tg.create_task(sub._queue.put(data))

    async def close(self) -> None:
        """Signal all subscribers that the stream has ended."""
        async with asyncio.TaskGroup() as tg:
            for sub in self.subscribers:
                tg.create_task(sub._queue.put(_CLOSED))

class Sub[T]:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | _Closed] = asyncio.Queue(10)
        self._closed = False

    def __aiter__(self) -> "Sub[T]":
        return self

    async def __anext__(self) -> T:
        if self._closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, _Closed):
            self._closed = True
            raise StopAsyncIteration
        return item
