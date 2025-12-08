import asyncio

class Pub[T]:
    def __init__(self) -> None:
        self.subscribers: list[Sub[T]] = []

    def sub(self) -> "Sub[T]":
        sub = Sub()
        self.subscribers.append(sub)
        return sub
    
    async def publish(self, data: T) -> None:
        async with asyncio.TaskGroup() as tg:
            for sub in self.subscribers:
                tg.create_task(sub.queue.put(data))

class Sub[T]:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[T] = asyncio.Queue(10)

    def __aiter__(self) -> "Sub[T]":
        return self

    async def __anext__(self) -> T:
        return await self.queue.get()
