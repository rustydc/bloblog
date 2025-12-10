"""Tests for node validation and execution."""
import asyncio
import pytest

from bloblog import run, validate_nodes
from test_utils import make_producer_node, make_consumer_node, make_transform_node


class TestValidateNodes:
    def test_valid_simple_pipeline(self):
        producer = make_producer_node("producer", ["msg"])
        consumer, _ = make_consumer_node("consumer", "producer_output", 1)
        
        # Should not raise
        validate_nodes([producer, consumer])

    def test_duplicate_output_channel_raises(self):
        producer1 = make_producer_node("producer", ["msg"])
        producer2 = make_producer_node("producer", ["msg"])  # Same output channel name
        
        with pytest.raises(ValueError, match="produced by multiple nodes"):
            validate_nodes([producer1, producer2])

    def test_missing_input_producer_raises(self):
        consumer, _ = make_consumer_node("consumer", "nonexistent_channel", 1)
        
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
        consumer, received = make_consumer_node("consumer", "producer_output", len(messages))

        await run([producer, consumer])

        assert received == messages

    @pytest.mark.asyncio
    async def test_producer_with_multiple_consumers(self):
        """Test one producer with multiple consumers (fan-out)."""
        messages = ["a", "b", "c"]
        producer = make_producer_node("producer", messages)
        consumer1, received1 = make_consumer_node("consumer1", "producer_output", len(messages))
        consumer2, received2 = make_consumer_node("consumer2", "producer_output", len(messages))

        await run([producer, consumer1, consumer2])

        assert received1 == messages
        assert received2 == messages

    @pytest.mark.asyncio
    async def test_transform_pipeline(self):
        """Test a producer -> transformer -> consumer pipeline."""
        messages = ["hello", "world"]
        producer = make_producer_node("producer", messages)
        transformer = make_transform_node(
            "transformer", 
            "producer_output", 
            "transformer_output",
            lambda s: s.upper(),
            len(messages)
        )
        consumer, received = make_consumer_node("consumer", "transformer_output", len(messages))

        await run([producer, transformer, consumer])

        assert received == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test pipeline with empty message list completes the producer."""
        messages = ["single"]
        producer = make_producer_node("producer", messages)
        consumer, received = make_consumer_node("consumer", "producer_output", len(messages))

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
        consumer, _ = make_consumer_node("consumer", "nonexistent_channel", 1)

        with pytest.raises(ValueError, match="has no producer node"):
            await run([consumer])
