"""Bloblog node graph execution and management."""
from __future__ import annotations

import asyncio
import inspect
import pickle
from collections.abc import Callable, Awaitable
from pathlib import Path
from typing import Annotated, Any, get_type_hints, get_origin, get_args

from .pubsub import In, Out
from .bloblog import BlobLogWriter
from .codecs import Codec


class _ChannelRuntime[T]:
    """Internal runtime channel linking publishers and subscribers."""
    def __init__(self, name: str, codec: Codec[T]) -> None:
        self.name = name
        self.codec = codec
        self.out = Out[T]()


def _parse_node_signature(
    node_fn: Callable,
) -> tuple[dict[str, str], dict[str, tuple[str, Codec]]]:
    """Parse a node function's signature to extract input/output channels.
    
    Args:
        node_fn: The node function or bound method to inspect.
    
    Returns:
        (inputs, outputs) where:
        - inputs maps param_name -> channel_name (no codec)
        - outputs maps param_name -> (channel_name, codec)
    
    Raises:
        ValueError: If signature is invalid or codecs are missing from outputs.
    """
    sig = inspect.signature(node_fn)
    hints = get_type_hints(node_fn, include_extras=True)
    
    inputs: dict[str, str] = {}
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
                f"must be Annotated[In[T], \"channel\"] or Annotated[Out[T], \"channel\", codec]"
            )
        
        # Parse Annotated[In[T] | Out[T], channel_name, codec?]
        args = get_args(hint)
        if len(args) < 2:
            raise ValueError(
                f"Parameter '{param_name}' must be Annotated[In[T], \"channel\"] "
                f"or Annotated[Out[T], \"channel\", codec]"
            )
        
        base_type = args[0]  # In[T] or Out[T]
        channel_name = args[1]  # "camera"
        origin = get_origin(base_type)
        
        if origin is In:
            # Inputs must have exactly 2 args (no codec)
            if len(args) != 2:
                raise ValueError(
                    f"Input parameter '{param_name}' must be Annotated[In[T], \"channel\"] "
                    f"(codec discovered from output/log)"
                )
            inputs[param_name] = channel_name
            
        elif origin is Out:
            # Outputs must have exactly 3 args (codec required)
            if len(args) != 3:
                raise ValueError(
                    f"Output parameter '{param_name}' must be Annotated[Out[T], \"channel\", codec]"
                )
            codec = args[2]
            if not isinstance(codec, Codec):
                raise ValueError(
                    f"Output parameter '{param_name}' codec must be a Codec instance, "
                    f"got {type(codec)}"
                )
            outputs[param_name] = (channel_name, codec)
            
        else:
            raise ValueError(
                f"Parameter '{param_name}' must be In[T] or Out[T], got {base_type}"
            )
    
    return inputs, outputs


def validate_nodes(
    nodes: list[Callable],
) -> None:
    """Validate that node inputs/outputs are properly connected.
    
    Args:
        nodes: List of node callables.
    
    Raises:
        ValueError: If validation fails.
    """
    output_channels: dict[str, Callable] = {}
    
    # Collect all outputs
    for node_fn in nodes:
        _, outputs = _parse_node_signature(node_fn)
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
        inputs, _ = _parse_node_signature(node_fn)
        for param_name, channel_name in inputs.items():
            if channel_name not in output_channels:
                raise ValueError(
                    f"Input channel '{channel_name}' in {node_fn.__name__}() "
                    f"has no producer node"
                )


async def _logging_task(channel: _ChannelRuntime, writer: BlobLogWriter) -> None:
    """Subscribe to a channel and log all messages to a bloblog file.
    
    Writes codec metadata as the first record for self-describing logs.
    """
    sub = channel.out.sub()
    write = writer.get_writer(channel.name)
    
    # Write codec metadata as first record
    codec_classname = channel.codec.get_qualified_name()
    codec_bytes = await asyncio.to_thread(pickle.dumps, channel.codec)
    
    # Encode as: [classname_length, classname_bytes, codec_length, codec_bytes]
    classname_encoded = codec_classname.encode('utf-8')
    metadata = (
        len(classname_encoded).to_bytes(4, 'little') +
        classname_encoded +
        len(codec_bytes).to_bytes(4, 'little') +
        codec_bytes
    )
    write(metadata)
    
    # Write regular data
    async for item in sub:
        data = await asyncio.to_thread(channel.codec.encode, item)
        write(data)


async def run_nodes(
    nodes: list[Callable[..., Awaitable[None]]],
    log_dir: Path | None = None,
) -> None:
    """Run a graph of nodes, optionally logging all channel data.
    
    Args:
        nodes: List of async callables (functions or bound methods).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name", codec_instance]
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
    
    # Validate nodes
    validate_nodes(nodes)
    
    # Parse all node signatures once and build runtime channels
    runtime_channels: dict[str, _ChannelRuntime] = {}
    node_info: dict[Callable, tuple[dict, dict, dict[str, Any]]] = {}  # (inputs, outputs, kwargs)
    
    for node_fn in nodes:
        inputs, outputs = _parse_node_signature(node_fn)
        
        # Create runtime channels for outputs
        for param_name, (channel_name, codec) in outputs.items():
            if channel_name not in runtime_channels:
                runtime_channels[channel_name] = _ChannelRuntime(channel_name, codec)
        
        node_info[node_fn] = (inputs, outputs, {})
    
    # Pre-build kwargs for all nodes to avoid race conditions
    # All subscriptions must be created BEFORE any node starts running
    for node_fn, (inputs, outputs, kwargs) in node_info.items():
        for param_name, channel_name in inputs.items():
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
