"""Benchmark optimized publish implementations."""

import asyncio

import pytest

from tinman.pubsub import In, Out


class OptimizedOut:
    """Out with optimized publish (sequential)."""

    def __init__(self) -> None:
        self.subscribers: list[In] = []

    def sub(self) -> In:
        sub: In = In()
        self.subscribers.append(sub)
        return sub

    async def publish(self, data) -> None:
        """Sequential publish - no gather overhead."""
        for sub in self.subscribers:
            await sub._queue.put(data)

    async def close(self) -> None:
        from tinman.pubsub import _CLOSED

        for sub in self.subscribers:
            await sub._queue.put(_CLOSED)


class OptimizedOutGenerator:
    """Out with optimized publish (generator expression)."""

    def __init__(self) -> None:
        self.subscribers: list[In] = []

    def sub(self) -> In:
        sub: In = In()
        self.subscribers.append(sub)
        return sub

    async def publish(self, data) -> None:
        """Use generator instead of list comp."""
        await asyncio.gather(*(sub._queue.put(data) for sub in self.subscribers))

    async def close(self) -> None:
        from tinman.pubsub import _CLOSED

        await asyncio.gather(*(sub._queue.put(_CLOSED) for sub in self.subscribers))


class OptimizedOutNoGather:
    """Out with optimized publish (no gather, concurrent via create_task)."""

    def __init__(self) -> None:
        self.subscribers: list[In] = []

    def sub(self) -> In:
        sub: In = In()
        self.subscribers.append(sub)
        return sub

    async def publish(self, data) -> None:
        """Create tasks without gather overhead."""
        tasks = [asyncio.create_task(sub._queue.put(data)) for sub in self.subscribers]
        for task in tasks:
            await task

    async def close(self) -> None:
        from tinman.pubsub import _CLOSED

        tasks = [asyncio.create_task(sub._queue.put(_CLOSED)) for sub in self.subscribers]
        for task in tasks:
            await task


class TestPublishOptimizations:
    """Compare different publish implementations."""

    @pytest.mark.benchmark(group="publish-optimization")
    def test_current_publish_list_comp(self, benchmark):
        """Current implementation with list comprehension."""

        async def run():
            out = Out()
            inp = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer():
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="publish-optimization")
    def test_optimized_sequential(self, benchmark):
        """Optimized: sequential put (no gather)."""

        async def run():
            out = OptimizedOut()
            inp = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer():
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="publish-optimization")
    def test_optimized_generator(self, benchmark):
        """Optimized: generator expression instead of list comp."""

        async def run():
            out = OptimizedOutGenerator()
            inp = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer():
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="publish-optimization")
    def test_optimized_create_task(self, benchmark):
        """Optimized: create_task without gather."""

        async def run():
            out = OptimizedOutNoGather()
            inp = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer():
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="publish-optimization-fanout")
    def test_current_fanout_3_subscribers(self, benchmark):
        """Current implementation with 3 subscribers."""

        async def run():
            out = Out()
            inp1 = out.sub()
            inp2 = out.sub()
            inp3 = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer(inp):
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer(inp1), consumer(inp2), consumer(inp3))

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="publish-optimization-fanout")
    def test_optimized_sequential_fanout_3(self, benchmark):
        """Optimized sequential with 3 subscribers."""

        async def run():
            out = OptimizedOut()
            inp1 = out.sub()
            inp2 = out.sub()
            inp3 = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer(inp):
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer(inp1), consumer(inp2), consumer(inp3))

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="publish-optimization-fanout")
    def test_optimized_generator_fanout_3(self, benchmark):
        """Optimized generator with 3 subscribers."""

        async def run():
            out = OptimizedOutGenerator()
            inp1 = out.sub()
            inp2 = out.sub()
            inp3 = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer(inp):
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer(inp1), consumer(inp2), consumer(inp3))

        benchmark(lambda: asyncio.run(run()))
