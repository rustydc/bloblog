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

    @pytest.mark.benchmark(group="pipeline-transform")
    def test_transform_pipeline(self, benchmark):
        """Benchmark producer -> transform -> consumer pipeline."""

        async def run_pipeline():
            codec = SimpleCodec()
            received = []

            async def producer(out: Annotated[Out[str], "input", codec]):
                for i in range(1000):
                    await out.publish(f"msg_{i}")

            async def transformer(
                inp: Annotated[In[str], "input"],
                out: Annotated[Out[str], "output", codec],
            ):
                async for msg in inp:
                    await out.publish(msg.upper())

            async def consumer(inp: Annotated[In[str], "output"]):
                async for msg in inp:
                    received.append(msg)

            await run([producer, transformer, consumer])
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

    @pytest.mark.benchmark(group="pipeline-multichannel")
    def test_multiple_channels(self, benchmark):
        """Benchmark pipeline with multiple channels."""

        async def run_pipeline():
            codec = SimpleCodec()
            results = {"a": [], "b": [], "c": []}

            async def producer_a(out: Annotated[Out[str], "chan_a", codec]):
                for i in range(500):
                    await out.publish(f"a_{i}")

            async def producer_b(out: Annotated[Out[str], "chan_b", codec]):
                for i in range(500):
                    await out.publish(f"b_{i}")

            async def producer_c(out: Annotated[Out[str], "chan_c", codec]):
                for i in range(500):
                    await out.publish(f"c_{i}")

            async def consumer_a(inp: Annotated[In[str], "chan_a"]):
                async for msg in inp:
                    results["a"].append(msg)

            async def consumer_b(inp: Annotated[In[str], "chan_b"]):
                async for msg in inp:
                    results["b"].append(msg)

            async def consumer_c(inp: Annotated[In[str], "chan_c"]):
                async for msg in inp:
                    results["c"].append(msg)

            await run([producer_a, producer_b, producer_c, consumer_a, consumer_b, consumer_c])
            return sum(len(v) for v in results.values())

        result = benchmark(lambda: asyncio.run(run_pipeline()))
        assert result == 1500

    @pytest.mark.benchmark(group="queue-backpressure")
    def test_slow_consumer(self, benchmark):
        """Benchmark pipeline with slow consumer (tests queue behavior)."""

        async def run_pipeline():
            codec = SimpleCodec()
            received = []

            async def fast_producer(out: Annotated[Out[str], "data", codec]):
                for i in range(100):
                    await out.publish(f"msg_{i}")

            async def slow_consumer(inp: Annotated[In[str], "data"]):
                async for msg in inp:
                    await asyncio.sleep(0.001)  # 1ms delay per message
                    received.append(msg)

            await run([fast_producer, slow_consumer])
            return len(received)

        result = benchmark(lambda: asyncio.run(run_pipeline()))
        assert result == 100

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

    @pytest.mark.benchmark(group="pipeline-comparison")
    def test_direct_queue_baseline(self, benchmark):
        """Baseline: direct asyncio.Queue without framework."""

        async def run_direct():
            queue = asyncio.Queue()
            count = [0]

            async def producer():
                for i in range(1000):
                    await queue.put(f"msg_{i}")
                await queue.put(None)  # Sentinel

            async def consumer():
                while True:
                    msg = await queue.get()
                    if msg is None:
                        break
                    count[0] += 1

            await asyncio.gather(producer(), consumer())
            return count[0]

        result = benchmark(lambda: asyncio.run(run_direct()))
        assert result == 1000

    @pytest.mark.benchmark(group="pipeline-comparison")
    def test_direct_queue_with_pickle(self, benchmark):
        """Baseline: direct asyncio.Queue with pickle (fair comparison)."""
        import pickle

        async def run_direct():
            queue = asyncio.Queue()
            count = [0]

            async def producer():
                for i in range(1000):
                    msg = f"msg_{i}"
                    # Serialize like the framework does
                    serialized = pickle.dumps(msg)
                    await queue.put(serialized)
                await queue.put(None)  # Sentinel

            async def consumer():
                while True:
                    msg = await queue.get()
                    if msg is None:
                        break
                    # Deserialize like the framework does
                    _ = pickle.loads(msg)
                    count[0] += 1

            await asyncio.gather(producer(), consumer())
            return count[0]

        result = benchmark(lambda: asyncio.run(run_direct()))
        assert result == 1000

    @pytest.mark.benchmark(group="pipeline-comparison")
    def test_framework_pure_overhead(self, benchmark):
        """Framework overhead with no-op binary codec (no serialization cost)."""

        async def run_pipeline():
            count = [0]

            # Use binary codec (no-op serialization)
            class BinaryCodec(Codec[bytes]):
                def encode(self, item: bytes) -> bytes:
                    return item

                def decode(self, data) -> bytes:
                    return bytes(data)

            codec = BinaryCodec()

            async def producer(out: Annotated[Out[bytes], "data", codec]):
                for i in range(1000):
                    await out.publish(b"test_message")

            async def consumer(inp: Annotated[In[bytes], "data"]):
                async for _msg in inp:
                    count[0] += 1

            await run([producer, consumer])
            return count[0]

        result = benchmark(lambda: asyncio.run(run_pipeline()))
        assert result == 1000

    @pytest.mark.benchmark(group="hot-path")
    def test_publish_subscribe_hot_path(self, benchmark):
        """Measure just the publish/subscribe overhead (no framework setup)."""

        async def measure_hot_path():
            # Setup once (not measured)
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

            # Hot path: just the message passing
            await asyncio.gather(producer(), consumer())
            return count[0]

        result = benchmark(lambda: asyncio.run(measure_hot_path()))
        assert result == 1000

    @pytest.mark.benchmark(group="hot-path")
    def test_direct_queue_hot_path(self, benchmark):
        """Measure raw asyncio.Queue for comparison."""

        async def measure_hot_path():
            # Setup once (not measured)
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

            # Hot path: just the message passing
            await asyncio.gather(producer(), consumer())
            return count[0]

        result = benchmark(lambda: asyncio.run(measure_hot_path()))
        assert result == 1000
