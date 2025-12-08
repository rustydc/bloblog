from __future__ import annotations

import asyncio
from collections.abc import Buffer
from abc import abstractmethod, ABC
from pathlib import Path
from typing import overload

from .pubsub import Pub, Sub
from .bloblog import BlobLogWriter


class Codec[T](ABC):
    @abstractmethod
    def encode(self, item: T) -> Buffer:
        ...
    
    @abstractmethod
    def decode(self, data: Buffer) -> T:
        ...


class Input[T]:
    """Descriptor for declaring a node input channel."""
    def __init__(self, name: str, codec: Codec[T]):
        self.name = name
        self.codec = codec
        self._attr_name: str = ""
    
    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = f"_input_{name}"
    
    @overload
    def __get__(self, obj: None, owner: type) -> Input[T]: ...
    @overload
    def __get__(self, obj: Node, owner: type) -> Sub[T]: ...
    
    def __get__(self, obj: Node | None, owner: type) -> Input[T] | Sub[T]:
        if obj is None:
            return self  # Class-level access returns the descriptor
        return getattr(obj, self._attr_name)
    
    def __set__(self, obj: Node, value: Sub[T]) -> None:
        setattr(obj, self._attr_name, value)


class Output[T]:
    """Descriptor for declaring a node output channel."""
    def __init__(self, name: str, codec: Codec[T]):
        self.name = name
        self.codec = codec
        self._attr_name: str = ""
    
    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = f"_output_{name}"
    
    @overload
    def __get__(self, obj: None, owner: type) -> Output[T]: ...
    @overload
    def __get__(self, obj: Node, owner: type) -> Pub[T]: ...
    
    def __get__(self, obj: Node | None, owner: type) -> Output[T] | Pub[T]:
        if obj is None:
            return self  # Class-level access returns the descriptor
        return getattr(obj, self._attr_name)
    
    def __set__(self, obj: Node, value: Pub[T]) -> None:
        setattr(obj, self._attr_name, value)


class Node(ABC):
    """Base class for processing nodes.
    
    Subclasses declare inputs/outputs as class attributes:
    
        class MyNode(Node):
            image: Input[Image] = Input("camera", ImageCodec())
            detections: Output[Detection] = Output("detections", DetectionCodec())
            
            async def run(self):
                async for img in self.image:
                    await self.detections.publish(detect(img))
    """
    
    def get_inputs(self) -> list[tuple[str, Input]]:
        """Return list of (attr_name, Input) for this node."""
        return [(name, attr) for name in dir(self.__class__) 
                if isinstance(attr := getattr(self.__class__, name), Input)]
    
    def get_outputs(self) -> list[tuple[str, Output]]:
        """Return list of (attr_name, Output) for this node."""
        return [(name, attr) for name in dir(self.__class__) 
                if isinstance(attr := getattr(self.__class__, name), Output)]

    @abstractmethod
    async def run(self) -> None:
        """Process data from inputs to outputs."""
        ...


class _Channel[T]:
    """Internal runtime channel linking publishers and subscribers."""
    def __init__(self, name: str, codec: Codec[T], pub: Pub[T]) -> None:
        self.name = name
        self.codec = codec
        self.pub = pub

    @classmethod
    def from_output(cls, out: Output[T]) -> _Channel[T]:
        return cls(out.name, out.codec, Pub[T]())


def validate_nodes(nodes: list[Node]) -> None:
    """Validate that node inputs/outputs are properly connected."""
    output_channels: dict[str, Node] = {}
    # Make sure each output channel is produced by exactly one node.
    for node in nodes:
        for _, out in node.get_outputs():
            if out.name in output_channels:
                raise ValueError(f"Output channel '{out.name}' produced by multiple nodes.")
            output_channels[out.name] = node

    # Make sure every input channel has a corresponding output channel.
    for node in nodes:
        for _, inp in node.get_inputs():
            if inp.name not in output_channels:
                raise ValueError(f"Input channel '{inp.name}' has no producer node.")


async def _logging_task(channel: _Channel, writer: BlobLogWriter) -> None:
    """Subscribe to a channel and log all messages to a bloblog file."""
    sub = channel.pub.sub()
    write = writer.get_writer(channel.name)
    async for item in sub:
        data = channel.codec.encode(item)
        write(bytes(data))


async def run_nodes(nodes: list[Node], log_dir: Path | None = None) -> None:
    """Run a graph of nodes, optionally logging all channel data.
    
    Args:
        nodes: List of nodes to run.
        log_dir: Directory to write log files for node outputs.
    """
    validate_nodes(nodes)

    # Build channels from all output declarations
    channels: dict[str, _Channel] = {}
    for node in nodes:
        for _, out in node.get_outputs():
            channels[out.name] = _Channel.from_output(out)

    writer = BlobLogWriter(log_dir) if log_dir else None

    async def run_node_and_close(node: Node) -> None:
        """Run a node and close its output channels when done."""
        try:
            await node.run()
        finally:
            for _, out in node.get_outputs():
                await channels[out.name].pub.close()

    async with asyncio.TaskGroup() as tg:
        # Set up logging tasks for all outputs
        if writer:
            for channel in channels.values():
                tg.create_task(_logging_task(channel, writer))

        # Wire up and run all nodes
        for node in nodes:
            for attr_name, inp in node.get_inputs():
                setattr(node, attr_name, channels[inp.name].pub.sub())
            for attr_name, out in node.get_outputs():
                setattr(node, attr_name, channels[out.name].pub)
            tg.create_task(run_node_and_close(node))

    if writer:
        await writer.close()
