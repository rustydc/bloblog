"""Bloblog node graph execution and management."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from .bloblog import BlobLogWriter
from .codecs import Codec, PickleCodec
from .playback import write_channel_encoded, _playback_task
from .pubsub import In, Out


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

    for param_name, _param in sig.parameters.items():
        if param_name == "self":
            continue

        if param_name not in hints:
            raise ValueError(f"Parameter '{param_name}' in {node_fn.__name__}() has no type hint")

        hint = hints[param_name]

        # Must be Annotated
        if get_origin(hint) is not Annotated:
            raise ValueError(
                f"Parameter '{param_name}' in {node_fn.__name__}() "
                f'must be Annotated[In[T], "channel"] or '
                f'Annotated[Out[T], "channel"] (or with explicit codec)'
            )

        # Parse Annotated[In[T] | Out[T], channel_name, codec?]
        args = get_args(hint)
        if len(args) < 2:
            raise ValueError(
                f"Parameter '{param_name}' must be Annotated[In[T], \"channel\"] "
                f'or Annotated[Out[T], "channel", codec]'
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
            # Outputs can have 2 args (use PickleCodec) or 3 args (explicit codec)
            if len(args) == 2:
                # No codec specified, use PickleCodec as default
                codec = PickleCodec()
            elif len(args) == 3:
                codec = args[2]
                if not isinstance(codec, Codec):
                    raise ValueError(
                        f"Output parameter '{param_name}' codec must be a Codec instance, "
                        f"got {type(codec)}"
                    )
            else:
                raise ValueError(
                    f"Output parameter '{param_name}' must be "
                    f'Annotated[Out[T], "channel"] or Annotated[Out[T], "channel", codec]'
                )
            outputs[param_name] = (channel_name, codec)

        else:
            raise ValueError(f"Parameter '{param_name}' must be In[T] or Out[T], got {base_type}")

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
        for _param_name, (channel_name, _codec) in outputs.items():
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
        for _param_name, channel_name in inputs.items():
            if channel_name not in output_channels:
                raise ValueError(
                    f"Input channel '{channel_name}' in {node_fn.__name__}() has no producer node"
                )


async def _logging_task(channel: _ChannelRuntime, writer: BlobLogWriter) -> None:
    """Subscribe to a channel and log all messages to a bloblog file.

    Uses write_channel_encoded to automatically handle codec metadata and encoding.
    """
    
    sub = channel.out.sub()
    await write_channel_encoded(channel.name, channel.codec, sub, writer)


async def run(
    nodes: list[Callable[..., Awaitable[None]]],
    log_dir: Path | None = None,
    playback_dir: Path | None = None,
    playback_speed: float = 0,
) -> None:
    """Run a graph of nodes, optionally logging and/or playing back channel data.

    Args:
        nodes: List of async callables (functions or bound methods).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name", codec_instance]
        log_dir: Optional directory to write log files for outputs.
        playback_dir: Optional directory containing .bloblog files to play back from.
                      If provided, any input channels not produced by live nodes
                      will be played back from logs.
        playback_speed: Playback speed multiplier. 0 = as fast as possible,
                        1.0 = real-time. Only used if playback_dir is set.

    Raises:
        TypeError: If nodes are not async callables.
        ValueError: If channel configuration is invalid.
        FileNotFoundError: If playback is enabled and required log files don't exist.
    """

    # Validate all are async callables
    for node in nodes:
        if not asyncio.iscoroutinefunction(node):
            raise TypeError(
                f"{node} must be an async function or method. "
                f"If using a class, pass instance.run not the instance itself."
            )

    # Parse all node signatures to find inputs and outputs
    all_inputs: set[str] = set()
    all_outputs: set[str] = set()
    node_info: dict[Callable, tuple[dict, dict, dict[str, Any]]] = {}  # (inputs, outputs, kwargs)

    for node_fn in nodes:
        inputs, outputs = _parse_node_signature(node_fn)
        
        for _param_name, channel_name in inputs.items():
            all_inputs.add(channel_name)
        
        for _param_name, (channel_name, _codec) in outputs.items():
            if channel_name in all_outputs:
                raise ValueError(
                    f"Output channel '{channel_name}' produced by multiple nodes"
                )
            all_outputs.add(channel_name)
        
        node_info[node_fn] = (inputs, outputs, {})
    
    # Find inputs that aren't produced by any node (need playback)
    missing_inputs = all_inputs - all_outputs
    
    # Setup playback if needed
    playback_channels: dict[str, tuple[Path, Out]] = {}
    if playback_dir and missing_inputs:
        # Validate that log files exist for all missing inputs
        for channel_name in missing_inputs:
            log_path = playback_dir / f"{channel_name}.bloblog"
            if not log_path.exists():
                raise FileNotFoundError(
                    f"No log file for channel '{channel_name}': {log_path}"
                )
            # Create output channel for playback
            out = Out()
            playback_channels[channel_name] = (log_path, out)
    elif missing_inputs:
        # No playback_dir but have missing inputs - error
        raise ValueError(
            f"Input channels have no producer nodes and no playback: {missing_inputs}"
        )
    
    # Build runtime channels - includes both live outputs and playback outputs
    runtime_channels: dict[str, _ChannelRuntime] = {}
    
    for node_fn in nodes:
        inputs, outputs, _ = node_info[node_fn]
        
        # Create runtime channels for node outputs
        for _param_name, (channel_name, codec) in outputs.items():
            if channel_name not in runtime_channels:
                runtime_channels[channel_name] = _ChannelRuntime(channel_name, codec)
    
    # Pre-build kwargs for all nodes to avoid race conditions
    # All subscriptions must be created BEFORE any node starts running
    for _node_fn, (inputs, outputs, kwargs) in node_info.items():
        for param_name, channel_name in inputs.items():
            # Get the Out from either runtime channels or playback channels
            if channel_name in runtime_channels:
                out = runtime_channels[channel_name].out
            elif channel_name in playback_channels:
                _, out = playback_channels[channel_name]
            else:
                raise ValueError(f"No source for input channel '{channel_name}'")
            kwargs[param_name] = out.sub()

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
            for _param_name, (channel_name, _) in outputs.items():
                await runtime_channels[channel_name].out.close()

    async with asyncio.TaskGroup() as tg:
        # Set up logging tasks for all outputs (but not playback channels)
        if writer:
            for channel_name, channel in runtime_channels.items():
                # Only log channels that are produced by nodes, not playback
                if channel_name not in playback_channels:
                    tg.create_task(_logging_task(channel, writer))

        # Start playback task if needed
        if playback_channels:
            tg.create_task(_playback_task(playback_channels, playback_speed))

        # Run all nodes
        for node_fn in nodes:
            tg.create_task(run_node_and_close(node_fn))

    if writer:
        await writer.close()
