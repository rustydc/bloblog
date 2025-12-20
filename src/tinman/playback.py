"""Playback transforms for replaying recorded data.

This module provides transforms for playing back recorded channel data
from ObLog files, with support for various playback speeds.

Example:
    >>> from tinman import Graph
    >>> from tinman.playback import with_playback
    >>> 
    >>> graph = Graph.of(consumer)
    >>> graph = (await with_playback(Path("logs")))(graph)
    >>> await graph.run()
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING

from .bloblog import amerge
from .oblog import Codec, ObLogReader
from .pubsub import Out
from .runtime import NodeSpec, get_node_specs
from .timer import Timer, ScaledTimer, FastForwardTimer, VirtualClock

if TYPE_CHECKING:
    from .launcher import Graph


async def create_playback_graph(
    nodes: list[Callable | NodeSpec],
    playback_dir: Path,
    speed: float = 1.0,
    clock: VirtualClock | None = None,
) -> tuple[list[Callable | NodeSpec], int | None]:
    """Create a graph with playback nodes injected for missing inputs.

    This function analyzes the given nodes to find which channels they need,
    checks which channels are missing (not produced by any node), and creates
    a playback node to provide those channels from recorded logs.

    Args:
        nodes: List of node callables to run.
        playback_dir: Directory containing .blog log files.
        speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed, 
               inf = fast-forward).
        clock: VirtualClock for fast-forward mode. Required if speed=inf.
               The playback node will advance this clock as it processes messages.

    Returns:
        Tuple of (graph, first_timestamp) where:
        - graph: List of nodes with playback nodes injected
        - first_timestamp: Timestamp (ns) of first message across all channels,
          or None if no playback needed

    Raises:
        FileNotFoundError: If a required log file doesn't exist.
        ValueError: If speed=inf but no clock provided.

    Example:
        >>> # Record some data
        >>> await run([producer, create_recording_node(Path("logs"), codecs)])
        >>> # Play back later (real-time)
        >>> graph, first_ts = await create_playback_graph([consumer], Path("logs"))
        >>> await run_nodes(graph)
        >>> # Fast-forward playback
        >>> clock = VirtualClock()
        >>> timer = FastForwardTimer(clock)
        >>> graph = await create_playback_graph([consumer], Path("logs"), speed=float('inf'), clock=clock)
        >>> await run_nodes(graph, timer=timer)
    """
    if speed == float('inf') and clock is None:
        raise ValueError("Fast-forward mode (speed=inf) requires a VirtualClock")

    # Parse nodes to find what channels are needed and what's provided
    specs = get_node_specs(nodes)
    all_inputs: set[str] = set()
    all_outputs: set[str] = set()

    for spec in specs:
        for _, (channel_name, _) in spec.inputs.items():
            all_inputs.add(channel_name)
        for _, (channel_name, _) in spec.outputs.items():
            all_outputs.add(channel_name)

    # Find channels that need playback
    missing_channels = all_inputs - all_outputs

    if not missing_channels:
        # No playback needed
        return list(nodes), None

    # Read codecs and first timestamps for all missing channels
    oblog = ObLogReader(playback_dir)
    channel_codecs: dict[str, Codec] = {}
    first_timestamp: int | None = None
    
    for channel in missing_channels:
        try:
            codec = await oblog.read_codec(channel)
            channel_codecs[channel] = codec
            # Get first timestamp for this channel
            ts = await oblog.first_timestamp(channel)
            if ts is not None and (first_timestamp is None or ts < first_timestamp):
                first_timestamp = ts
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No log file for channel '{channel}' in {playback_dir}"
            )

    # Create a single playback node that outputs all missing channels
    # Capture oblog in closure so we reuse the same reader
    async def playback_node(**outputs: Out) -> None:
        """Playback node that reads from logs and publishes to output channels."""
        # Build list of channels for index mapping
        channel_list = list(channel_codecs.items())

        # Set up readers for all channels using the shared oblog reader
        readers = [oblog.read_channel(channel_name) for channel_name, _codec in channel_list]

        # Merge all streams and publish with timing
        start_time: int | None = None
        start_real: int | None = None
        first_message = True

        async for source_idx, timestamp, item in amerge(*readers):
            channel_name, _ = channel_list[source_idx]
            out = outputs[channel_name]

            if speed == float('inf'):
                # Fast-forward mode: advance virtual clock
                assert clock is not None
                if first_message:
                    # Initialize clock to first message timestamp
                    clock._time = timestamp
                    first_message = False
                else:
                    # Advance clock, waking any timers in between
                    await clock.advance_to(timestamp)
                
                # Publish and yield to let consumer process at this timestamp
                await out.publish(item)
                await asyncio.sleep(0)
            else:
                # Speed-controlled playback with real delays
                if speed > 0:
                    if start_time is None:
                        start_time = timestamp
                        start_real = time_ns()
                        first_message = False
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

        # After all messages, flush any remaining timers
        if speed == float('inf') and clock is not None:
            # Yield to let consumers process their messages and schedule timers
            await asyncio.sleep(0)
            # Wake all pending timers
            await clock.flush()

    # Create NodeSpec for the playback node with output annotations
    playback_outputs = {
        channel_name: (channel_name, codec)
        for channel_name, codec in channel_codecs.items()
    }
    playback_spec = NodeSpec(
        node_fn=playback_node,
        inputs={},
        outputs=playback_outputs,
        all_channels_param=None,
        timer_param=None,
        name="playback_node",
    )

    # Return playback node + original nodes, and first timestamp
    return [playback_spec, *nodes], first_timestamp


def with_playback(
    playback_dir: Path,
    speed: float = float('inf'),
    use_virtual_time_logs: bool = False,
) -> Callable[["Graph"], "Graph"]:
    """Inject playback nodes for missing inputs.
    
    Analyzes the graph to find which channels are needed but not produced,
    then creates playback nodes to provide those channels from recorded logs.
    
    Note: This transform must be applied before Graph.run() is called, as it
    needs to read log metadata synchronously. If you're already in an async
    context, use create_playback_graph() directly.
    
    Args:
        playback_dir: Directory containing .blog log files.
        speed: Playback speed multiplier:
               - float('inf') (default): Fast-forward (deterministic, ASAP)
               - 1.0: Realtime (respects original timestamps)
               - 2.0: Double speed
        use_virtual_time_logs: If True, Python log records use playback time.
        
    Returns:
        A transform function that adds playback to a graph.
        
    Example:
        >>> graph = Graph.of(consumer)
        >>> graph = with_playback(Path("logs"))(graph)
        >>> await graph.run()
    """
    def transform(g: "Graph") -> "Graph":
        # Create clock for fast-forward mode
        clock: VirtualClock | None = None
        if speed == float('inf'):
            clock = VirtualClock()
        
        # Run async playback graph creation synchronously
        # This is safe because we're not yet in an event loop
        playback_nodes, first_timestamp = asyncio.run(
            create_playback_graph(g.nodes, playback_dir, speed=speed, clock=clock)
        )
        
        # Update graph nodes (playback node is prepended by create_playback_graph)
        g.nodes = list(playback_nodes)
        
        # Create timer based on speed
        if speed == float('inf'):
            assert clock is not None
            if first_timestamp is not None:
                clock._time = first_timestamp
            g.timer = FastForwardTimer(clock)
        else:
            g.timer = ScaledTimer(speed, start_time=first_timestamp)
        
        # Handle virtual time logs
        if use_virtual_time_logs and g.timer is not None:
            from .logging import install_timer_log_factory
            install_timer_log_factory(g.timer)
        
        return g
    
    return transform
