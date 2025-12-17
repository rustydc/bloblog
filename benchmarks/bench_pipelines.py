"""End-to-end benchmarks for complete node pipelines."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import pytest

from tinman import In, Out, run
from tinman.oblog import Codec


class SimpleCodec(Codec[str]):
    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")

    def decode(self, data) -> str:
        return bytes(data).decode("utf-8")


class TestPipelinePerformance:
    """Benchmark complete node pipelines."""

    @pytest.mark.benchmark(group="pipeline-simple")
    def test_simple_producer_consumer(self, benchmark):
        """Benchmark simple producer -> consumer pipeline."""

        async def run_pipeline():
            codec = SimpleCodec()
            received = []

            async def producer(out: Annotated[Out[str], "messages", codec]):
                for i in range(1000):
                    await out.publish(f"message_{i}")

            async def consumer(inp: Annotated[In[str], "messages"]):
                async for msg in inp:
                    received.append(msg)

            await run([producer, consumer])
            return len(received)

        result = benchmark(lambda: asyncio.run(run_pipeline()))
        assert result == 1000

    @pytest.mark.benchmark(group="pipeline-fanout")
    def test_fanout_pipeline(self, benchmark):
        """Benchmark one producer with multiple consumers."""

        async def run_pipeline():
            codec = SimpleCodec()
            received1 = []
            received2 = []
            received3 = []

            async def producer(out: Annotated[Out[str], "data", codec]):
                for i in range(1000):
                    await out.publish(f"msg_{i}")

            async def consumer1(inp: Annotated[In[str], "data"]):
                async for msg in inp:
                    received1.append(msg)

            async def consumer2(inp: Annotated[In[str], "data"]):
                async for msg in inp:
                    received2.append(msg)

            async def consumer3(inp: Annotated[In[str], "data"]):
                async for msg in inp:
                    received3.append(msg)

            await run([producer, consumer1, consumer2, consumer3])
            return len(received1) + len(received2) + len(received3)

        result = benchmark(lambda: asyncio.run(run_pipeline()))
        assert result == 3000  # 1000 messages to each of 3 consumers

    @pytest.mark.benchmark(group="pipeline-logging")
    def test_pipeline_with_logging(self, benchmark):
        """Benchmark pipeline with logging enabled."""

        async def run_pipeline():
            with TemporaryDirectory() as tmpdir:
                codec = SimpleCodec()
                received = []

                async def producer(out: Annotated[Out[str], "logged", codec]):
                    for i in range(1000):
                        await out.publish(f"msg_{i}")

                async def consumer(inp: Annotated[In[str], "logged"]):
                    async for msg in inp:
                        received.append(msg)

                await run([producer, consumer], log_dir=Path(tmpdir))
                return len(received)

        result = benchmark(lambda: asyncio.run(run_pipeline()))
        assert result == 1000

    @pytest.mark.benchmark(group="pipeline-overhead")
    def test_framework_overhead(self, benchmark):
        """Measure framework overhead vs direct async communication."""

        async def run_with_framework():
            codec = SimpleCodec()
            count = [0]

            async def producer(out: Annotated[Out[str], "data", codec]):
                for i in range(1000):
                    await out.publish(f"msg_{i}")

            async def consumer(inp: Annotated[In[str], "data"]):
                async for _msg in inp:
                    count[0] += 1

            await run([producer, consumer])
            return count[0]

        result = benchmark(lambda: asyncio.run(run_with_framework()))
        assert result == 1000

    @pytest.mark.benchmark(group="hot-path")
    def test_publish_subscribe_hot_path(self, benchmark):
        """Measure just the publish/subscribe overhead (no framework setup)."""

        async def measure_hot_path():
            out = Out()
            inp = out.sub()
            count = [0]

            async def producer():
                for i in range(1000):
                    await out.publish(f"msg_{i}")
                await out.close()

            async def consumer():
                async for _msg in inp:
                    count[0] += 1

            await asyncio.gather(producer(), consumer())
            return count[0]

        result = benchmark(lambda: asyncio.run(measure_hot_path()))
        assert result == 1000

    @pytest.mark.benchmark(group="hot-path")
    def test_direct_queue_hot_path(self, benchmark):
        """Measure raw asyncio.Queue for comparison."""

        async def measure_hot_path():
            queue = asyncio.Queue()
            count = [0]

            async def producer():
                for i in range(1000):
                    await queue.put(f"msg_{i}")
                await queue.put(None)

            async def consumer():
                while True:
                    msg = await queue.get()
                    if msg is None:
                        break
                    count[0] += 1

            await asyncio.gather(producer(), consumer())
            return count[0]

        result = benchmark(lambda: asyncio.run(measure_hot_path()))
        assert result == 1000
