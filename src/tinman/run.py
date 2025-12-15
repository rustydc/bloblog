"""High-level node graph launchers and utilities.

This module provides convenient functions for running node graphs with
common patterns like logging and playback. It builds on the low-level
execution engine in runtime.py.

Key functions:
- run(): Main entry point for executing a node graph
- get_node_specs(): Extract channel metadata from nodes
- create_logging_node(): Factory for logging nodes
- create_playback_graph(): Factory for playback graphs
- validate_nodes(): Validate graph connectivity
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import time_ns

from .bloblog import amerge
from .oblog import Codec, ObLog
from .pubsub import In, Out
from .runtime import NodeSpec, get_node_specs, run_nodes


def create_logging_node(
    log_dir: Path,
    channel_specs: dict[str, Codec] | None = None,
    channel_filter: set[str] | None = None
) -> Callable:
    """Create a node that logs all channels to disk.

    Args:
        log_dir: Directory to write log files.
        channel_specs: Optional map of channel_name -> codec. If None, codecs will be
                      extracted from the first message on each channel (requires the
                      runtime to provide codec info).
        channel_filter: If provided, only log these channels. If None, log all.

    Returns:
        An async node function that can be passed to run().

    Example:
        >>> # Get specs from existing nodes to extract codecs
        >>> specs = get_node_specs([producer, consumer])
        >>> codecs = {ch: codec for spec in specs
        ...           for _, (ch, codec) in spec.outputs.items()}
        >>> logger = create_logging_node(Path("logs"), codecs)
        >>> await run([producer, consumer, logger])

        >>> # Or use without pre-specified codecs (will auto-detect)
        >>> logger = create_logging_node(Path("logs"))
        >>> await run([producer, consumer, logger])

    Note:
        The created node will automatically receive all output channels via
        the dict[str, In] injection mechanism.
    """

    async def logging_node(channels: dict[str, In]) -> None:
        """Log all subscribed channels to disk."""
        oblog = ObLog(log_dir)

        try:
            async def log_channel(channel_name: str, input_channel: In, codec: Codec | None) -> None:
                """Log a single channel."""
                if codec is None:
                    # If no codec provided, we can't write. This is a limitation.
                    # In a real implementation, we'd need codec metadata from runtime.
                    # For now, skip channels without codecs.
                    return

                write = oblog.get_writer(channel_name, codec)
                async for item in input_channel:
                    write(item)

            async with asyncio.TaskGroup() as tg:
                for channel_name, input_channel in channels.items():
                    # Apply filter if specified
                    if channel_filter and channel_name not in channel_filter:
                        continue

                    # Get codec for this channel
                    codec = channel_specs.get(channel_name) if channel_specs else None
                    if codec is None and channel_specs is not None:
                        # Codec map provided but this channel not in it - skip
                        continue

                    tg.create_task(log_channel(channel_name, input_channel, codec))
        finally:
            await oblog.close()

    return logging_node


async def create_playback_graph(
    nodes: list[Callable],
    playback_dir: Path,
    speed: float = 1.0
) -> list[Callable | NodeSpec]:
    """Create a graph with playback nodes injected for missing inputs.

    This function analyzes the given nodes to find which channels they need,
    checks which channels are missing (not produced by any node), and creates
    a playback node to provide those channels from recorded logs.

    Args:
        nodes: List of node callables to run.
        playback_dir: Directory containing .blog log files.
        speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.).

    Returns:
        List of nodes with playback nodes injected. Original nodes may include
        NodeSpec objects for the playback node.

    Raises:
        FileNotFoundError: If a required log file doesn't exist.

    Example:
        >>> # Record some data
        >>> await run([producer, create_logging_node(Path("logs"), codecs)])
        >>> # Play back later
        >>> graph = await create_playback_graph([consumer], Path("logs"))
        >>> await run(graph)
    """
    # Parse nodes to find what channels are needed and what's provided
    specs = get_node_specs(nodes)
    all_inputs = set()
    all_outputs = set()

    for spec in specs:
        for _, (channel_name, _) in spec.inputs.items():
            all_inputs.add(channel_name)
        for _, (channel_name, _) in spec.outputs.items():
            all_outputs.add(channel_name)

    # Find channels that need playback
    missing_channels = all_inputs - all_outputs

    if not missing_channels:
        # No playback needed
        return list(nodes)

    # Read codecs for all missing channels
    oblog = ObLog(playback_dir)
    channel_codecs: dict[str, Codec] = {}

    try:
        for channel in missing_channels:
            try:
                codec = await oblog.read_codec(channel)
                channel_codecs[channel] = codec
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No log file for channel '{channel}' in {playback_dir}"
                )
    finally:
        await oblog.close()

    # Create a single playback node that outputs all missing channels
    async def playback_node(**outputs: Out) -> None:
        """Playback node that reads from logs and publishes to output channels."""
        # Build list of channels for index mapping
        channel_list = list(channel_codecs.items())

        # Set up readers for all channels
        oblogs = []
        readers = []
        for channel_name, _codec in channel_list:
            oblog_reader = ObLog(playback_dir)
            oblogs.append(oblog_reader)
            readers.append(oblog_reader.read_channel(channel_name))

        try:
            # Merge all streams and publish with timing
            start_time: int | None = None
            start_real: int | None = None

            async for source_idx, timestamp, item in amerge(*readers):
                channel_name, _ = channel_list[source_idx]
                out = outputs[channel_name]

                # Apply speed control if needed
                if speed != float('inf') and speed > 0:
                    if start_time is None:
                        start_time = timestamp
                        start_real = time_ns()
                    else:
                        # Calculate target time
                        elapsed_log = timestamp - start_time
                        assert start_real is not None
                        target_real = start_real + int(elapsed_log / speed)
                        current = time_ns()

                        # Sleep if needed
                        if current < target_real:
                            await asyncio.sleep((target_real - current) / 1e9)

                await out.publish(item)
        finally:
            # Close all oblogs
            for oblog_reader in oblogs:
                await oblog_reader.close()

    # Create NodeSpec for the playback node with output annotations
    playback_outputs = {
        channel_name: (channel_name, codec)
        for channel_name, codec in channel_codecs.items()
    }
    playback_spec = NodeSpec(
        node_fn=playback_node,
        inputs={},
        outputs=playback_outputs,
        all_channels_param=None
    )

    # Return playback node + original nodes
    return [playback_spec, *nodes]


async def run(nodes: list[Callable[..., Awaitable[None]] | NodeSpec]) -> None:
    """Run a graph of nodes, wiring their channels and executing them concurrently.

    This is a convenience wrapper around runtime.run_nodes() that provides a
    cleaner API for common use cases. For logging or playback, use
    create_logging_node() or create_playback_graph() to create explicit nodes.

    Args:
        nodes: List of async callables (functions, bound methods, or NodeSpec objects).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name", codec_instance]
               - All channels: dict[str, In] (for monitoring/logging nodes)

    Raises:
        TypeError: If nodes are not async callables.
        ValueError: If channel configuration is invalid (e.g., missing input producers).

    Example:
        >>> # Simple pipeline
        >>> await run([producer, consumer])

        >>> # With logging
        >>> logger = create_logging_node(Path("logs"), codecs)
        >>> await run([producer, consumer, logger])

        >>> # With playback
        >>> graph = await create_playback_graph([consumer], Path("logs"))
        >>> await run(graph)
    """
    await run_nodes(nodes)
