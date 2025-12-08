from .node import NodeSpec, ChannelSpec, Codec, Node
from .pubsub import Pub, Sub
from .bloblog import BlobLogWriter
from pathlib import Path
import asyncio

def validate_nodes(nodes: list[NodeSpec]) -> None:
    output_channels = {}
    # Make sure each output channel is produced by exactly one node.
    for node in nodes:
        for ch in node.outputs:
            if ch.name in output_channels:
                raise ValueError(f"Output channel '{ch.name}' produced by multiple nodes.")
            output_channels[ch.name] = node

    # Make sure every input channel has a corresponding output channel.
    for node in nodes:
        for ch in node.inputs:
            if ch.name not in output_channels:
                raise ValueError(f"Input channel '{ch.name}' has no producer node.")

class Channel[T]:
    def __init__(self, name: str, codec: Codec[T], pub: Pub[T]) -> None:
        self.name = name
        self.codec = codec
        self.pub = pub

    @classmethod
    def from_spec(cls, spec: ChannelSpec[T]) -> "Channel[T]":
        pub = Pub[T]()
        return cls(spec.name, spec.codec, pub)


async def _logging_task(channel: Channel, writer: BlobLogWriter) -> None:
    """Subscribe to a channel and log all messages to a bloblog file."""
    sub = channel.pub.sub()
    write = writer.get_writer(channel.name)
    async for item in sub:
        data = channel.codec.encode(item)
        write(bytes(data))


async def run_nodes(nodes: list[Node], log_dir: Path | None = None) -> None:
    validate_nodes([node.get_spec() for node in nodes])

    channels = {spec.name: Channel.from_spec(spec)
                for node in nodes
                for spec in node.get_spec().outputs}

    writer = BlobLogWriter(log_dir) if log_dir else None

    async with asyncio.TaskGroup() as tg:
        # Set up logging tasks for each output channel
        if writer:
            for channel in channels.values():
                tg.create_task(_logging_task(channel, writer))

        for node in nodes:
            node_spec = node.get_spec()
            tg.create_task(node.process([channels[ch.name].pub.sub() for ch in node_spec.inputs],
                                        [channels[ch.name].pub for ch in node_spec.outputs]))

    if writer:
        await writer.close()
