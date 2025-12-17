"""Micro-benchmarks to understand where the Out/In overhead comes from."""

import asyncio

import pytest

from tinman.pubsub import In, Out


class TestPubSubMicroBenchmarks:
    """Drill down into the exact overhead sources."""

    @pytest.mark.benchmark(group="micro")
    def test_raw_queue_baseline(self, benchmark):
        """Raw asyncio.Queue baseline."""

        async def run():
            queue = asyncio.Queue()

            async def producer():
                for i in range(100):
                    await queue.put(i)
                await queue.put(None)

            async def consumer():
                count = 0
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="micro")
    def test_out_in_overhead(self, benchmark):
        """Out/In wrapper overhead."""

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

    @pytest.mark.benchmark(group="micro")
    def test_asyncio_gather_overhead(self, benchmark):
        """Cost of asyncio.gather per message."""

        async def run():
            queue = asyncio.Queue()

            async def producer():
                for i in range(100):
                    # Simulate Out.publish using gather
                    await asyncio.gather(queue.put(i))
                await queue.put(None)

            async def consumer():
                count = 0
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="micro")
    def test_list_comprehension_overhead(self, benchmark):
        """Cost of list comprehension in publish."""

        async def run():
            queue = asyncio.Queue()
            subscribers = [queue]  # Simulate Out.subscribers

            async def producer():
                for i in range(100):
                    # Simulate Out.publish pattern
                    await asyncio.gather(*[sub.put(i) for sub in subscribers])
                await queue.put(None)

            async def consumer():
                count = 0
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="micro")
    def test_isinstance_check_overhead(self, benchmark):
        """Cost of isinstance check in __anext__."""

        async def run():
            queue = asyncio.Queue()

            async def producer():
                for i in range(100):
                    await queue.put(i)
                await queue.put(None)

            async def consumer():
                count = 0
                while True:
                    item = await queue.get()
                    # Simulate In.__anext__ isinstance check
                    if isinstance(item, type(None)):
                        break
                    count += 1
                return count

            await asyncio.gather(producer(), consumer())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="micro")
    def test_async_for_overhead(self, benchmark):
        """Cost of async for vs while True."""

        async def run():
            out = Out()
            inp = out.sub()

            async def producer():
                for i in range(100):
                    await out.publish(i)
                await out.close()

            async def consumer_async_for():
                count = 0
                async for _ in inp:
                    count += 1
                return count

            await asyncio.gather(producer(), consumer_async_for())

        benchmark(lambda: asyncio.run(run()))

    @pytest.mark.benchmark(group="micro")
    def test_multiple_subscribers_overhead(self, benchmark):
        """Cost with multiple subscribers (fan-out)."""

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
