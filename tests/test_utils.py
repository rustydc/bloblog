"""Shared test utilities for node testing."""
from collections.abc import Buffer, Callable

from bloblog import Codec, Node, Input, Output


class StringCodec(Codec[str]):
    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")

    def decode(self, data: Buffer) -> str:
        return bytes(data).decode("utf-8")


def make_producer_node(name: str, messages: list[str]) -> Node:
    """Create a producer node that outputs messages."""
    
    class ProducerNode(Node):
        output: Output[str] = Output(f"{name}_output", StringCodec())
        
        async def run(self) -> None:
            for msg in messages:
                await self.output.publish(msg)
    
    return ProducerNode()


def make_consumer_node(name: str, input_channel: str, expected_count: int) -> tuple[Node, list[str]]:
    """Create a consumer node that collects messages."""
    received: list[str] = []
    
    class ConsumerNode(Node):
        input: Input[str] = Input(input_channel, StringCodec())
        
        async def run(self) -> None:
            async for msg in self.input:
                received.append(msg)
                if len(received) >= expected_count:
                    break
    
    return ConsumerNode(), received


def make_transform_node(name: str, input_channel: str, output_channel: str,
                        transform: Callable[[str], str], expected_count: int) -> Node:
    """Create a transform node that processes messages."""
    
    class TransformNode(Node):
        input: Input[str] = Input(input_channel, StringCodec())
        output: Output[str] = Output(output_channel, StringCodec())
        
        async def run(self) -> None:
            count = 0
            async for msg in self.input:
                await self.output.publish(transform(msg))
                count += 1
                if count >= expected_count:
                    break
    
    return TransformNode()
