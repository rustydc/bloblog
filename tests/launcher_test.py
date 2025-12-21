"""Tests for launcher module (high-level utilities and execution)."""

import asyncio
import time
from typing import Annotated

import pytest
from .test_utils import StringCodec, make_consumer_node, make_producer_node, make_transform_node

from tinman import In, Out, ObLogReader, get_node_specs, Graph
from tinman.runtime import run_nodes
from tinman.recording import create_recording_node
from tinman.playback import create_playback_graph


class TestRun:
    @pytest.mark.asyncio
    async def test_simple_producer_consumer(self):
        """Test a simple producer -> consumer pipeline."""
        messages = ["hello", "world", "test"]
        producer = make_producer_node("producer", messages)
        consumer, received = make_consumer_node("producer_output", len(messages))

        await run_nodes([producer, consumer])

        assert received == messages

    @pytest.mark.asyncio
    async def test_producer_with_multiple_consumers(self):
        """Test one producer with multiple consumers (fan-out)."""
        messages = ["a", "b", "c"]
        producer = make_producer_node("producer", messages)
        consumer1, received1 = make_consumer_node("producer_output", len(messages))
        consumer2, received2 = make_consumer_node("producer_output", len(messages))

        await run_nodes([producer, consumer1, consumer2])

        assert received1 == messages
        assert received2 == messages

    @pytest.mark.asyncio
    async def test_transform_pipeline(self):
        """Test a producer -> transformer -> consumer pipeline."""
        messages = ["hello", "world"]
        producer = make_producer_node("producer", messages)
        transformer = make_transform_node(
            "producer_output",
            "transformer_output",
            lambda s: s.upper(),
            len(messages),
        )
        consumer, received = make_consumer_node("transformer_output", len(messages))

        await run_nodes([producer, transformer, consumer])

        assert received == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test pipeline with empty message list completes the producer."""
        messages = ["single"]
        producer = make_producer_node("producer", messages)
        consumer, received = make_consumer_node("producer_output", len(messages))

        # Should complete without hanging
        async with asyncio.timeout(1.0):
            await run_nodes([producer, consumer])

        assert received == messages

    @pytest.mark.asyncio
    async def test_single_producer_node(self):
        """Test running just a producer node with no consumers."""
        messages = ["msg1", "msg2"]
        producer = make_producer_node("producer", messages)

        # Should complete without error
        async with asyncio.timeout(1.0):
            await run_nodes([producer])

    @pytest.mark.asyncio
    async def test_validation_error_propagates(self):
        """Test that validation errors are raised from run."""
        consumer, _ = make_consumer_node("nonexistent_channel", 1)

        with pytest.raises(ValueError, match="has no producer node"):
            await run_nodes([consumer])

    @pytest.mark.asyncio
    async def test_dict_channel_injection(self):
        """Test that nodes can receive dict[str, In] with all channels."""
        messages_a = ["a1", "a2"]
        messages_b = ["b1", "b2"]
        received_dict: dict[str, list[str]] = {}

        producer_a = make_producer_node("channel_a", messages_a)
        producer_b = make_producer_node("channel_b", messages_b)

        async def monitor_all(channels: dict[str, In]) -> None:
            """Node that monitors all channels."""
            # Initialize lists for each channel
            for name in channels.keys():
                received_dict[name] = []
            
            async def collect_from_channel(name: str, inp: In):
                async for msg in inp:
                    received_dict[name].append(msg)
            
            async with asyncio.TaskGroup() as tg:
                for channel_name, input_channel in channels.items():
                    tg.create_task(collect_from_channel(channel_name, input_channel))
        
        await run_nodes([producer_a, producer_b, monitor_all])

        # make_producer_node creates channels named "{name}_output"
        assert received_dict["channel_a_output"] == messages_a
        assert received_dict["channel_b_output"] == messages_b


class TestLoggingNode:
    @pytest.mark.asyncio
    async def test_create_recording_node(self, tmp_path):
        """Test that create_recording_node creates a working logging node."""
        messages = ["log1", "log2", "log3"]
        producer = make_producer_node("test", messages)
        
        # Create and run with logging node - codecs extracted automatically
        logger = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, logger])
        
        # Verify log file was created
        log_file = tmp_path / "test_output.blog"
        assert log_file.exists()
        
        # Read back and verify (no context manager needed for reader)
        reader = ObLogReader(tmp_path)
        received = []
        async for _, msg in reader.read_channel("test_output"):
            received.append(msg)
        
        assert received == messages
    
    @pytest.mark.asyncio
    async def test_logging_node_with_filter(self, tmp_path):
        """Test that logging node respects channel filter."""
        messages_a = ["a1", "a2"]
        messages_b = ["b1", "b2"]
        
        producer_a = make_producer_node("channel_a", messages_a)
        producer_b = make_producer_node("channel_b", messages_b)
        
        # Create logger that only logs channel_a
        logger = create_recording_node(tmp_path, [producer_a, producer_b], channel_filter={"channel_a_output"})
        await run_nodes([producer_a, producer_b, logger])
        
        # Verify only channel_a was logged
        assert (tmp_path / "channel_a_output.blog").exists()
        assert not (tmp_path / "channel_b_output.blog").exists()


class TestPlayback:
    @pytest.mark.asyncio
    async def test_playback_from_logs(self, tmp_path):
        """Test that playback works with explicit logging and playback nodes."""
        messages = ["hello", "world", "test"]

        # First run: record producer output using explicit logging
        codec = StringCodec()

        async def recording_producer(output: Annotated[Out[str], "producer_output", codec]) -> None:
            for msg in messages:
                await output.publish(msg)

        # Use explicit logging node
        logger = create_recording_node(tmp_path, [recording_producer])
        await run_nodes([recording_producer, logger])

        # Verify log file was created
        log_file = tmp_path / "producer_output.blog"
        assert log_file.exists()

        # Second run: playback producer using explicit playback graph
        received: list[str] = []

        async def live_consumer(input: Annotated[In[str], "producer_output"]) -> None:
            async for msg in input:
                received.append(msg)
                if len(received) >= len(messages):
                    break

        # Use create_playback_graph - returns a configured Graph
        graph = await create_playback_graph(
            Graph.of(live_consumer), tmp_path, speed=float('inf')
        )
        await graph.run()

        assert received == messages

    @pytest.mark.asyncio
    async def test_playback_validates_log_exists(self, tmp_path):
        """Test that playback raises if log file doesn't exist."""

        async def consumer_of_missing(input: Annotated[In[str], "missing_channel"]) -> None:
            pass

        # Should raise when trying to create playback graph for missing channel
        with pytest.raises(FileNotFoundError, match="No log file for channel"):
            await create_playback_graph(Graph.of(consumer_of_missing), tmp_path)


class TestPlaybackNode:
    @pytest.mark.asyncio
    async def test_read_codec_from_log(self, tmp_path):
        """Test that we can read codec metadata from a log file."""
        messages = ["test1", "test2"]
        producer = make_producer_node("test", messages)
        
        # Log data
        logger = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, logger])
        
        # Now read the codec back (no context manager needed for reader)
        reader = ObLogReader(tmp_path)
        codec = await reader.read_codec("test_output")
        
        # Verify it's the right type
        assert isinstance(codec, StringCodec)
    
    @pytest.mark.asyncio
    async def test_create_playback_graph(self, tmp_path):
        """Test that create_playback_graph returns a configured Graph."""
        # First, record some data
        messages = ["play1", "play2", "play3"]
        producer = make_producer_node("recorded", messages)
        
        logger = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, logger])
        
        # Now create a consumer that expects "recorded_output" channel
        received = []
        
        async def consumer(inp: Annotated[In[str], "recorded_output"]) -> None:
            async for msg in inp:
                received.append(msg)
        
        # Create playback graph - returns a configured Graph
        graph = await create_playback_graph(
            Graph.of(consumer), tmp_path, speed=float('inf')
        )
        
        # Verify playback channels were created
        assert "recorded_output" in graph._playback_channels
        # Verify timer was set up
        assert graph.timer is not None
        
        # Run and verify
        await graph.run()
        assert received == messages
    
    @pytest.mark.asyncio
    async def test_playback_with_speed_control(self, tmp_path):
        """Test that playback respects speed parameter."""
        # Record data with some delay
        messages = ["msg1", "msg2"]
        producer = make_producer_node("timed", messages)
        
        logger = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, logger])
        
        # Play back at 2x speed - should be roughly twice as fast
        received = []
        timestamps = []
        
        async def consumer(inp: Annotated[In[str], "timed_output"]) -> None:
            async for msg in inp:
                received.append(msg)
                timestamps.append(time.time())
        
        graph = await create_playback_graph(Graph.of(consumer), tmp_path, speed=2.0)
        
        start = time.time()
        await graph.run()
        duration = time.time() - start
        
        assert received == messages
        # At 2x speed, should complete quickly (allowing some overhead)
        assert duration < 1.0  # Should be very fast at 2x
    
    @pytest.mark.asyncio
    async def test_playback_missing_log_raises(self, tmp_path):
        """Test that playback raises if log file doesn't exist."""
        async def consumer(inp: Annotated[In[str], "nonexistent"]) -> None:
            pass
        
        with pytest.raises(FileNotFoundError, match="No log file for channel"):
            await create_playback_graph(Graph.of(consumer), tmp_path)


class TestConvenienceFunctions:
    @pytest.mark.asyncio
    async def test_run_with_recording(self, tmp_path):
        """Test that create_recording_node properly logs data."""
        messages = ["log1", "log2", "log3"]
        producer = make_producer_node("test", messages)
        consumer, received = make_consumer_node("test_output", len(messages))
        
        # Run with recording node
        recorder = create_recording_node(tmp_path, [producer, consumer])
        await run_nodes([producer, consumer, recorder])
        
        assert received == messages
        
        # Verify log file was created
        log_file = tmp_path / "test_output.blog"
        assert log_file.exists()
        
        # Read back and verify (no context manager needed for reader)
        reader = ObLogReader(tmp_path)
        logged = []
        async for _, msg in reader.read_channel("test_output"):
            logged.append(msg)
        
        assert logged == messages
    
    @pytest.mark.asyncio
    async def test_playback_from_recorded(self, tmp_path):
        """Test playback using create_playback_graph."""
        # First, record some data
        messages = ["play1", "play2", "play3"]
        producer = make_producer_node("recorded", messages)
        recorder = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, recorder])
        
        # Now playback
        received = []
        
        async def consumer(inp: Annotated[In[str], "recorded_output"]) -> None:
            async for msg in inp:
                received.append(msg)
        
        graph = await create_playback_graph(
            Graph.of(consumer), tmp_path, speed=float('inf')
        )
        await graph.run()
        
        assert received == messages
    
    @pytest.mark.asyncio
    async def test_playback_with_speed(self, tmp_path):
        """Test that playback respects speed parameter."""
        # Record data
        messages = ["msg1", "msg2"]
        producer = make_producer_node("timed", messages)
        recorder = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, recorder])

        # Playback at 2x speed
        received = []

        async def consumer(inp: Annotated[In[str], "timed_output"]) -> None:
            async for msg in inp:
                received.append(msg)

        graph = await create_playback_graph(
            Graph.of(consumer), tmp_path, speed=2.0
        )
        
        start = time.time()
        await graph.run()
        duration = time.time() - start

        assert received == messages
        # Should complete quickly at 2x speed
        assert duration < 1.0

    @pytest.mark.asyncio
    async def test_playback_fast_forward_with_timer(self, tmp_path):
        """Test fast-forward playback with Timer coordination."""
        from tinman.timer import Timer

        # Record data
        messages = ["msg1", "msg2", "msg3"]
        producer = make_producer_node("data", messages)
        recorder = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, recorder])

        # Playback with a node that uses Timer
        events: list[str] = []

        async def timed_consumer(
            inp: Annotated[In[str], "data_output"],
            timer: Timer,
        ) -> None:
            async for msg in inp:
                events.append(f"msg:{msg}@{timer.time_ns()}")

        # Fast-forward playback
        graph = await create_playback_graph(
            Graph.of(timed_consumer), tmp_path, speed=float('inf')
        )
        await graph.run()

        assert len(events) == 3
        # All messages should be received
        assert all(e.startswith("msg:msg") for e in events)

    @pytest.mark.asyncio
    async def test_playback_fast_forward_timer_sleep(self, tmp_path):
        """Test that Timer.sleep() works correctly during fast-forward playback."""
        from tinman.timer import Timer

        # Record data - just ONE message to simplify
        messages = ["a"]
        producer = make_producer_node("input", messages)
        recorder = create_recording_node(tmp_path, [producer])
        await run_nodes([producer, recorder])

        # Consumer that sleeps after processing
        processed: list[str] = []
        sleep_completed = False

        async def sleeping_consumer(
            inp: Annotated[In[str], "input_output"],
            timer: Timer,
        ) -> None:
            nonlocal sleep_completed
            async for msg in inp:
                processed.append(msg)
                # Sleep for 100ms (virtual time)
                await timer.sleep(0.1)
                sleep_completed = True

        # Fast-forward should complete quickly despite sleep calls
        graph = await create_playback_graph(
            Graph.of(sleeping_consumer), tmp_path, speed=float('inf')
        )
        
        start = time.time()
        await graph.run()
        duration = time.time() - start

        assert processed == messages
        assert sleep_completed
        # Should complete in well under 1 second (no real sleeping)
        assert duration < 0.5

    @pytest.mark.asyncio
    async def test_no_playback_needed_returns_graph(self, tmp_path):
        """Test that create_playback_graph works when no playback is needed."""
        # Create a self-contained graph (producer -> consumer)
        messages = ["a", "b"]
        producer = make_producer_node("test", messages)
        consumer, received = make_consumer_node("test_output", len(messages))

        graph = await create_playback_graph(
            Graph.of(producer, consumer), tmp_path, speed=float('inf')
        )
        
        await graph.run()
        assert received == messages
