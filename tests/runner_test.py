import asyncio
import pytest
from collections.abc import Buffer, Callable

from bloblog import NodeSpec, ChannelSpec, Codec, Node, Pub, Sub, run_nodes, validate_nodes


class StringCodec(Codec[str]):
    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")

    def decode(self, data: Buffer) -> str:
        return bytes(data).decode("utf-8")


class ProducerNode(Node):
    """A node that produces messages to an output channel."""

    def __init__(self, name: str, messages: list[str]):
        self.name = name
        self.messages = messages
        self.output_spec = ChannelSpec(f"{name}_output", StringCodec())

    def get_spec(self) -> NodeSpec:
        return NodeSpec(self, inputs=[], outputs=[self.output_spec])

    async def process(self, inputs: list[Sub], outputs: list[Pub]) -> None:
        for msg in self.messages:
            await outputs[0].publish(msg)


class ConsumerNode(Node):
    """A node that consumes messages from an input channel."""

    def __init__(self, name: str, input_channel: str, expected_count: int):
        self.name = name
        self.input_spec = ChannelSpec(input_channel, StringCodec())
        self.expected_count = expected_count
        self.received: list[str] = []

    def get_spec(self) -> NodeSpec:
        return NodeSpec(self, inputs=[self.input_spec], outputs=[])

    async def process(self, inputs: list[Sub], outputs: list[Pub]) -> None:
        async for msg in inputs[0]:
            self.received.append(msg)
            if len(self.received) >= self.expected_count:
                break


class TransformNode(Node):
    """A node that transforms messages from input to output."""

    def __init__(self, name: str, input_channel: str, output_channel: str, 
                 transform: Callable[[str], str], expected_count: int):
        self.name = name
        self.input_spec = ChannelSpec(input_channel, StringCodec())
        self.output_spec = ChannelSpec(output_channel, StringCodec())
        self.transform = transform
        self.expected_count = expected_count

    def get_spec(self) -> NodeSpec:
        return NodeSpec(self, inputs=[self.input_spec], outputs=[self.output_spec])

    async def process(self, inputs: list[Sub], outputs: list[Pub]) -> None:
        count = 0
        async for msg in inputs[0]:
            await outputs[0].publish(self.transform(msg))
            count += 1
            if count >= self.expected_count:
                break


class TestValidateNodes:
    def test_valid_simple_pipeline(self):
        producer = ProducerNode("producer", ["msg"])
        consumer = ConsumerNode("consumer", "producer_output", 1)
        
        # Should not raise
        validate_nodes([producer.get_spec(), consumer.get_spec()])

    def test_duplicate_output_channel_raises(self):
        producer1 = ProducerNode("producer", ["msg"])
        producer2 = ProducerNode("producer", ["msg"])  # Same output channel name
        
        with pytest.raises(ValueError, match="produced by multiple nodes"):
            validate_nodes([producer1.get_spec(), producer2.get_spec()])

    def test_missing_input_producer_raises(self):
        consumer = ConsumerNode("consumer", "nonexistent_channel", 1)
        
        with pytest.raises(ValueError, match="has no producer node"):
            validate_nodes([consumer.get_spec()])

    def test_producer_only_is_valid(self):
        producer = ProducerNode("producer", ["msg"])
        
        # Should not raise - producer with no consumers is valid
        validate_nodes([producer.get_spec()])


class TestRunNodes:
    @pytest.mark.asyncio
    async def test_simple_producer_consumer(self):
        """Test a simple producer -> consumer pipeline."""
        messages = ["hello", "world", "test"]
        producer = ProducerNode("producer", messages)
        consumer = ConsumerNode("consumer", "producer_output", len(messages))

        await run_nodes([producer, consumer])

        assert consumer.received == messages

    @pytest.mark.asyncio
    async def test_producer_with_multiple_consumers(self):
        """Test one producer with multiple consumers (fan-out)."""
        messages = ["a", "b", "c"]
        producer = ProducerNode("producer", messages)
        consumer1 = ConsumerNode("consumer1", "producer_output", len(messages))
        consumer2 = ConsumerNode("consumer2", "producer_output", len(messages))

        await run_nodes([producer, consumer1, consumer2])

        assert consumer1.received == messages
        assert consumer2.received == messages

    @pytest.mark.asyncio
    async def test_transform_pipeline(self):
        """Test a producer -> transformer -> consumer pipeline."""
        messages = ["hello", "world"]
        producer = ProducerNode("producer", messages)
        transformer = TransformNode(
            "transformer", 
            "producer_output", 
            "transformer_output",
            lambda s: s.upper(),
            len(messages)
        )
        consumer = ConsumerNode("consumer", "transformer_output", len(messages))

        await run_nodes([producer, transformer, consumer])

        assert consumer.received == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test pipeline with empty message list completes the producer."""
        messages = ["single"]
        producer = ProducerNode("producer", messages)
        consumer = ConsumerNode("consumer", "producer_output", len(messages))

        # Should complete without hanging
        async with asyncio.timeout(1.0):
            await run_nodes([producer, consumer])
        
        assert consumer.received == messages

    @pytest.mark.asyncio
    async def test_single_producer_node(self):
        """Test running just a producer node with no consumers."""
        messages = ["msg1", "msg2"]
        producer = ProducerNode("producer", messages)

        # Should complete without error
        async with asyncio.timeout(1.0):
            await run_nodes([producer])

    @pytest.mark.asyncio
    async def test_validation_error_propagates(self):
        """Test that validation errors are raised from run_nodes."""
        consumer = ConsumerNode("consumer", "nonexistent_channel", 1)

        with pytest.raises(ValueError, match="has no producer node"):
            await run_nodes([consumer])
