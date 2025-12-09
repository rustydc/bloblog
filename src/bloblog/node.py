"""Bloblog node graph execution and management."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Buffer, Callable, Awaitable
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, get_type_hints, get_origin, get_args

from .pubsub import In, Out
from .bloblog import BlobLogWriter


class Codec[T](ABC):
    """Abstract base class for encoding/decoding channel data."""
    @abstractmethod
    def encode(self, item: T) -> Buffer:
        ...
    
    @abstractmethod
    def decode(self, data: Buffer) -> T:
        ...


class _ChannelRuntime[T]:
    """Internal runtime channel linking publishers and subscribers."""
    def __init__(self, name: str, codec: Codec[T]) -> None:
        self.name = name
        self.codec = codec
        self.out = Out[T]()


def _parse_node_signature(
    node_fn: Callable,
    codec_registry: dict[str, Codec],
) -> tuple[dict[str, tuple[str, Codec]], dict[str, tuple[str, Codec]]]:
    """Parse a node function's signature to extract input/output channels.
    
    Args:
        node_fn: The node function or bound method to inspect.
        codec_registry: Registry of channel codecs.
    
    Returns:
        (inputs, outputs) where each is a dict mapping:
        param_name -> (channel_name, codec)
    
    Raises:
        ValueError: If signature is invalid or codecs are missing.
    """
    sig = inspect.signature(node_fn)
    hints = get_type_hints(node_fn, include_extras=True)
    
    inputs: dict[str, tuple[str, Codec]] = {}
    outputs: dict[str, tuple[str, Codec]] = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        if param_name not in hints:
            raise ValueError(
                f"Parameter '{param_name}' in {node_fn.__name__}() has no type hint"
            )
        
        hint = hints[param_name]
        
        # Must be Annotated
        if get_origin(hint) is not Annotated:
            raise ValueError(
                f"Parameter '{param_name}' in {node_fn.__name__}() "
                f"must be Annotated[In[T], ...] or Annotated[Out[T], ...]"
            )
        
        # Parse Annotated[In[T] | Out[T], channel_name, codec?]
        args = get_args(hint)
        if len(args) < 2:
            raise ValueError(
                f"Parameter '{param_name}' must be Annotated[In[T], channel_name] "
                f"or Annotated[In[T], channel_name, codec]"
            )
        
        base_type = args[0]  # In[T] or Out[T]
        channel_name = args[1]  # "camera"
        
        # Codec can be in annotation (3rd arg) or registry
        codec: Codec
        if len(args) >= 3:
            codec = args[2]
        elif channel_name in codec_registry:
            codec = codec_registry[channel_name]
        else:
            raise ValueError(
                f"Channel '{channel_name}' used by {node_fn.__name__}() "
                f"needs a codec. Provide it in the annotation or channels config."
            )
        
        origin = get_origin(base_type)
        
        if origin is In:
            inputs[param_name] = (channel_name, codec)
        elif origin is Out:
            outputs[param_name] = (channel_name, codec)
        else:
            raise ValueError(
                f"Parameter '{param_name}' must be In[T] or Out[T], got {base_type}"
            )
    
    return inputs, outputs


def validate_nodes(
    nodes: list[Callable],
    codec_registry: dict[str, Codec],
) -> None:
    """Validate that node inputs/outputs are properly connected.
    
    Args:
        nodes: List of node callables.
        codec_registry: Registry of channel codecs.
    
    Raises:
        ValueError: If validation fails.
    """
    output_channels: dict[str, Callable] = {}
    
    # Collect all outputs
    for node_fn in nodes:
        _, outputs = _parse_node_signature(node_fn, codec_registry)
        for param_name, (channel_name, codec) in outputs.items():
            if channel_name in output_channels:
                raise ValueError(
                    f"Output channel '{channel_name}' produced by multiple nodes: "
                    f"{output_channels[channel_name].__name__} and "
                    f"{node_fn.__name__}"
                )
            output_channels[channel_name] = node_fn
    
    # Check all inputs have corresponding outputs
    for node_fn in nodes:
        inputs, _ = _parse_node_signature(node_fn, codec_registry)
        for param_name, (channel_name, codec) in inputs.items():
            if channel_name not in output_channels:
                raise ValueError(
                    f"Input channel '{channel_name}' in {node_fn.__name__}() "
                    f"has no producer node"
                )


async def _logging_task(channel: _ChannelRuntime, writer: BlobLogWriter) -> None:
    """Subscribe to a channel and log all messages to a bloblog file."""
    sub = channel.out.sub()
    write = writer.get_writer(channel.name)
    async for item in sub:
        data = await asyncio.to_thread(channel.codec.encode, item)
        write(data)


async def run_nodes(
    nodes: list[Callable[..., Awaitable[None]]],
    channels: dict[str, Codec] | None = None,
    log_dir: Path | None = None,
) -> None:
    """Run a graph of nodes, optionally logging all channel data.
    
    Args:
        nodes: List of async callables (functions or bound methods).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name"]
        channels: Optional dict mapping channel names to codecs.
                  If None, codecs must be in node annotations.
        log_dir: Optional directory to write log files for outputs.
    
    Raises:
        TypeError: If nodes are not async callables.
        ValueError: If channel configuration is invalid.
    """
    
    # Validate all are async callables
    for node in nodes:
        if not asyncio.iscoroutinefunction(node):
            raise TypeError(
                f"{node} must be an async function or method. "
                f"If using a class, pass instance.run not the instance itself."
            )
    
    # Build codec registry
    codec_registry: dict[str, Codec] = channels if channels else {}
    
    # Validate nodes
    validate_nodes(nodes, codec_registry)
    
    # Parse all node signatures once and build runtime channels
    runtime_channels: dict[str, _ChannelRuntime] = {}
    node_info: dict[Callable, tuple[dict, dict, dict[str, Any]]] = {}  # (inputs, outputs, kwargs)
    
    for node_fn in nodes:
        inputs, outputs = _parse_node_signature(node_fn, codec_registry)
        
        # Create runtime channels for outputs
        for param_name, (channel_name, codec) in outputs.items():
            if channel_name not in runtime_channels:
                runtime_channels[channel_name] = _ChannelRuntime(channel_name, codec)
        
        node_info[node_fn] = (inputs, outputs, {})
    
    # Pre-build kwargs for all nodes to avoid race conditions
    # All subscriptions must be created BEFORE any node starts running
    for node_fn, (inputs, outputs, kwargs) in node_info.items():
        for param_name, (channel_name, _) in inputs.items():
            kwargs[param_name] = runtime_channels[channel_name].out.sub()
        
        for param_name, (channel_name, _) in outputs.items():
            kwargs[param_name] = runtime_channels[channel_name].out
    
    writer = BlobLogWriter(log_dir) if log_dir else None
    
    async def run_node_and_close(node_fn: Callable) -> None:
        """Run a node and close its output channels when done."""
        _, outputs, kwargs = node_info[node_fn]
        try:
            await node_fn(**kwargs)
        finally:
            # Close output channels
            for param_name, (channel_name, _) in outputs.items():
                await runtime_channels[channel_name].out.close()
    
    async with asyncio.TaskGroup() as tg:
        # Set up logging tasks for all outputs
        if writer:
            for channel in runtime_channels.values():
                tg.create_task(_logging_task(channel, writer))
        
        # Run all nodes
        for node_fn in nodes:
            tg.create_task(run_node_and_close(node_fn))
    
    if writer:
        await writer.close()
