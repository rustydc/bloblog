"""Benchmark to test behavior with slow consumers."""

import asyncio

import pytest

from tinman.pubsub import In, Out


class SequentialOut:
    """Sequential publish."""

    def __init__(self) -> None:
        self.subscribers: list[In] = []

    def sub(self) -> In:
        sub: In = In()
        self.subscribers.append(sub)
        return sub

    async def publish(self, data) -> None:
        for sub in self.subscribers:
            await sub._queue.put(data)

    async def close(self) -> None:
        from tinman.pubsub import _CLOSED
        for sub in self.subscribers:
            await sub._queue.put(_CLOSED)


class ConcurrentOut:
    """Concurrent publish with gather."""

    def __init__(self) -> None:
        self.subscribers: list[In] = []

    def sub(self) -> In:
        sub: In = In()
        self.subscribers.append(sub)
        return sub

    async def publish(self, data) -> None:
        await asyncio.gather(*(sub._queue.put(data) for sub in self.subscribers))

    async def close(self) -> None:
        from tinman.pubsub import _CLOSED
        await asyncio.gather(*(sub._queue.put(_CLOSED) for sub in self.subscribers))


class TestSlowConsumerBehavior:
    """Test how sequential vs concurrent handles slow consumers."""

    @pytest.mark.benchmark(group="slow-consumer")
    def test_sequential_with_slow_consumer(self, benchmark):
        """Sequential: one slow consumer blocks producer."""

        async def run():
            out = SequentialOut()
            fast_inp = out.sub()
            slow_inp = out.sub()

            async def producer():
                for i in range(20):
                    await out.publish(i)
                await out.close()

            async def fast_consumer():
                """Processes immediately."""
                count = 0
                async for _ in fast_inp:
                    count += 1
                return count

            async def slow_consumer():
                """Processes with 1ms delay."""
                count = 0
                async for _ in slow_inp:
                    await asyncio.sleep(0.001)  # 1ms delay
                    count += 1
                return count

            await asyncio.gather(producer(), fast_consumer(), slow_consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="slow-consumer")
    def test_concurrent_with_slow_consumer(self, benchmark):
        """Concurrent: slow consumer doesn't block producer."""

        async def run():
            out = ConcurrentOut()
            fast_inp = out.sub()
            slow_inp = out.sub()

            async def producer():
                for i in range(20):
                    await out.publish(i)
                await out.close()

            async def fast_consumer():
                count = 0
                async for _ in fast_inp:
                    count += 1
                return count

            async def slow_consumer():
                count = 0
                async for _ in slow_inp:
                    await asyncio.sleep(0.001)
                    count += 1
                return count

            await asyncio.gather(producer(), fast_consumer(), slow_consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="queue-full")
    def test_sequential_queue_full(self, benchmark):
        """Sequential: what happens when queue fills up?"""

        async def run():
            out = SequentialOut()
            fast_inp = out.sub()
            blocked_inp = out.sub()

            async def producer():
                # Try to send 20 messages (queue capacity is 10)
                for i in range(20):
                    await out.publish(i)
                await out.close()

            async def fast_consumer():
                """Consumes normally."""
                count = 0
                async for _ in fast_inp:
                    count += 1
                return count

            async def blocked_consumer():
                """Never consumes - queue should fill up."""
                # Wait a bit, then start consuming
                await asyncio.sleep(0.01)  # 10ms delay before starting
                count = 0
                async for _ in blocked_inp:
                    count += 1
                return count

            await asyncio.gather(producer(), fast_consumer(), blocked_consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="queue-full")
    def test_concurrent_queue_full(self, benchmark):
        """Concurrent: what happens when queue fills up?"""

        async def run():
            out = ConcurrentOut()
            fast_inp = out.sub()
            blocked_inp = out.sub()

            async def producer():
                for i in range(20):
                    await out.publish(i)
                await out.close()

            async def fast_consumer():
                count = 0
                async for _ in fast_inp:
                    count += 1
                return count

            async def blocked_consumer():
                await asyncio.sleep(0.01)
                count = 0
                async for _ in blocked_inp:
                    count += 1
                return count

            await asyncio.gather(producer(), fast_consumer(), blocked_consumer())

        benchmark(lambda: asyncio.run(run()))
