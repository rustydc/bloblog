"""Recording transforms for data logging.

This module provides transforms for recording channel data to disk
in the ObLog binary format for later playback.

Example:
    >>> from tinman import Graph
    >>> from tinman.recording import with_recording
    >>> 
    >>> graph = Graph.of(producer, consumer)
    >>> graph = with_recording(Path("logs"))(graph)
    >>> await graph.run()
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .oblog import Codec, ObLogWriter
from .pubsub import In
from .runtime import NodeSpec, get_node_specs

if TYPE_CHECKING:
    from .launcher import Graph


def create_recording_node(
    log_dir: Path,
    nodes: list[Callable[..., Awaitable[None]] | NodeSpec] | None = None,
    channel_filter: set[str] | None = None
) -> NodeSpec:
    """Create a daemon node that records all channels to disk.

    Args:
        log_dir: Directory to write log files.
        nodes: Optional list of nodes to extract codecs from. If None, no channels
              will be recorded (useful if you want to add nodes later).
        channel_filter: If provided, only record these channels. If None, record all.

    Returns:
        A NodeSpec (daemon) that can be passed to Graph.of() or run_nodes().

    Example:
        >>> # Extract codecs from nodes automatically
        >>> recorder = create_recording_node(Path("logs"), [producer, consumer])
        >>> await run_nodes([producer, consumer, recorder])

        >>> # Filter specific channels
        >>> recorder = create_recording_node(Path("logs"), [producer], channel_filter={"important"})

    Note:
        The created node will automatically receive all output channels via
        the dict[str, In] injection mechanism. It's marked as a daemon so it
        won't block shutdown when other nodes complete.
    """
    
    # Extract codecs from nodes if provided
    channel_specs: dict[str, Codec] | None = None
    if nodes is not None:
        specs = get_node_specs(nodes)
        channel_specs = {
            channel_name: codec
            for spec in specs
            for _, (channel_name, codec) in spec.outputs.items()
        }

    async def recording_node(channels: dict[str, In]) -> None:
        """Record all subscribed channels to disk."""
        async with ObLogWriter(log_dir) as oblog:
            async def record_channel(channel_name: str, input_channel: In, codec: Codec | None) -> None:
                """Record a single channel."""
                if codec is None:
                    # If no codec provided, we can't write.
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

                    tg.create_task(record_channel(channel_name, input_channel, codec))

    return NodeSpec(
        node_fn=recording_node,
        inputs={},
        outputs={},
        all_channels_param="channels",
        daemon=True,
        name="recording_node",
    )


def with_recording(
    log_dir: Path,
    channel_filter: set[str] | None = None,
) -> Callable[[Graph], Graph]:
    """Add a recording node that saves all channel data to disk.
    
    Args:
        log_dir: Directory to write log files (.blog format).
        channel_filter: If provided, only record these channels.
        
    Returns:
        A transform function that adds recording to a graph.
        
    Example:
        >>> graph = Graph.of(producer, consumer)
        >>> graph = with_recording(Path("logs"))(graph)
        >>> await graph.run()
        
        >>> # Record only specific channels
        >>> graph = with_recording(Path("logs"), channel_filter={"important"})(graph)
    """
    def transform(g: Graph) -> Graph:
        g.nodes.append(create_recording_node(log_dir, g.nodes, channel_filter))
        return g
    return transform
