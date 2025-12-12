"""Tests for runner module (node validation, execution, and playback)."""

import asyncio
from typing import Annotated

import pytest
from test_utils import StringCodec, make_consumer_node, make_producer_node, make_transform_node

from tinman import In, Out, run, validate_nodes


class TestValidateNodes:
    def test_valid_simple_pipeline(self):
        producer = make_producer_node("producer", ["msg"])
        consumer, _ = make_consumer_node("producer_output", 1)

        # Should not raise
        validate_nodes([producer, consumer])

    def test_duplicate_output_channel_raises(self):
        producer1 = make_producer_node("producer", ["msg"])
        producer2 = make_producer_node("producer", ["msg"])  # Same output channel name

        with pytest.raises(ValueError, match="produced by multiple nodes"):
            validate_nodes([producer1, producer2])

    def test_missing_input_producer_raises(self):
        consumer, _ = make_consumer_node("nonexistent_channel", 1)

        with pytest.raises(ValueError, match="has no producer node"):
            validate_nodes([consumer])

    def test_producer_only_is_valid(self):
        producer = make_producer_node("producer", ["msg"])

        # Should not raise - producer with no consumers is valid
        validate_nodes([producer])


class TestRunNodes:
    @pytest.mark.asyncio
    async def test_simple_producer_consumer(self):
        """Test a simple producer -> consumer pipeline."""
        messages = ["hello", "world", "test"]
        producer = make_producer_node("producer", messages)
        consumer, received = make_consumer_node("producer_output", len(messages))

        await run([producer, consumer])

        assert received == messages

    @pytest.mark.asyncio
    async def test_producer_with_multiple_consumers(self):
        """Test one producer with multiple consumers (fan-out)."""
        messages = ["a", "b", "c"]
        producer = make_producer_node("producer", messages)
        consumer1, received1 = make_consumer_node("producer_output", len(messages))
        consumer2, received2 = make_consumer_node("producer_output", len(messages))

        await run([producer, consumer1, consumer2])

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

        await run([producer, transformer, consumer])

        assert received == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test pipeline with empty message list completes the producer."""
        messages = ["single"]
        producer = make_producer_node("producer", messages)
        consumer, received = make_consumer_node("producer_output", len(messages))

        # Should complete without hanging
        async with asyncio.timeout(1.0):
            await run([producer, consumer])

        assert received == messages

    @pytest.mark.asyncio
    async def test_single_producer_node(self):
        """Test running just a producer node with no consumers."""
        messages = ["msg1", "msg2"]
        producer = make_producer_node("producer", messages)

        # Should complete without error
        async with asyncio.timeout(1.0):
            await run([producer])

    @pytest.mark.asyncio
    async def test_validation_error_propagates(self):
        """Test that validation errors are raised from run_nodes."""
        consumer, _ = make_consumer_node("nonexistent_channel", 1)

        with pytest.raises(ValueError, match="no producer nodes and no playback"):
            await run([consumer])


class TestPlayback:
    @pytest.mark.asyncio
    async def test_playback_from_logs(self, tmp_path):
        """Test that non-live nodes play back from logs."""
        messages = ["hello", "world", "test"]

        # First run: record producer output
        codec = StringCodec()

        async def recording_producer(output: Annotated[Out[str], "producer_output", codec]) -> None:
            for msg in messages:
                await output.publish(msg)

        await run([recording_producer], log_dir=tmp_path)

        # Verify log file was created
        log_file = tmp_path / "producer_output.blog"
        assert log_file.exists()

        # Second run: playback producer (implicitly), run consumer live
        received: list[str] = []

        async def live_consumer(input: Annotated[In[str], "producer_output"]) -> None:
            async for msg in input:
                received.append(msg)
                if len(received) >= len(messages):
                    break

        # run with playback adds log players for producer_output and runs everything
        output_dir = tmp_path / "output"
        await run([live_consumer], playback_dir=tmp_path, log_dir=output_dir)

        assert received == messages

    @pytest.mark.asyncio
    async def test_playback_validates_log_exists(self, tmp_path):
        """Test that run with playback raises if log file doesn't exist."""

        async def consumer_of_missing(input: Annotated[In[str], "missing_channel"]) -> None:
            pass

        with pytest.raises(FileNotFoundError, match="No log file for channel"):
            await run([consumer_of_missing], playback_dir=tmp_path)
