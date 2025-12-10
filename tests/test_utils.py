"""Shared test utilities for node testing."""
from collections.abc import Buffer, Callable, Awaitable
from typing import Annotated

from bloblog import Codec, In, Out


class StringCodec(Codec[str]):
    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")

    def decode(self, data: Buffer) -> str:
        return bytes(data).decode("utf-8")


def make_producer_node(name: str, messages: list[str]) -> Callable[..., Awaitable[None]]:
    """Create a producer node that outputs messages."""
    
    async def producer_node(
        output: Annotated[Out[str], f"{name}_output", StringCodec()],
    ):
        for msg in messages:
            await output.publish(msg)
    
    return producer_node


def make_consumer_node(name: str, input_channel: str, expected_count: int) -> tuple[Callable[..., Awaitable[None]], list[str]]:
    """Create a consumer node that collects messages."""
    received: list[str] = []
    
    async def consumer_node(
        input: Annotated[In[str], input_channel],
    ):
        async for msg in input:
            received.append(msg)
            if len(received) >= expected_count:
                break
    
    return consumer_node, received


def make_transform_node(name: str, input_channel: str, output_channel: str,
                        transform, expected_count: int) -> Callable[..., Awaitable[None]]:
    """Create a transform node that processes messages."""
    
    async def transform_node(
        input: Annotated[In[str], input_channel],
        output: Annotated[Out[str], output_channel, StringCodec()],
    ):
        count = 0
        async for msg in input:
            await output.publish(transform(msg))
            count += 1
            if count >= expected_count:
                break
    
    return transform_node
