"""High-level launchers and utilities for node graphs.

This module provides convenient functions for running node graphs with
common patterns like logging and playback. It builds on the low-level
execution engine in runtime.py.

Key functions:
- run(): Convenience wrapper for executing a node graph
- create_logging_node(): Factory for logging nodes
- create_playback_graph(): Factory for playback graphs

Core functions are in runtime.py:
- get_node_specs(): Extract channel metadata from nodes
- validate_nodes(): Validate graph connectivity
- run_nodes(): Low-level execution engine
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import time_ns

from .bloblog import amerge
from .oblog import Codec, ObLogWriter, ObLogReader
from .pubsub import In, Out
from .runtime import NodeSpec, get_node_specs, run_nodes
from .timer import Timer, ScaledTimer, FastForwardTimer, VirtualClock


def create_logging_node(
    log_dir: Path,
    nodes: list[Callable[..., Awaitable[None]] | NodeSpec] | None = None,
    channel_filter: set[str] | None = None
) -> NodeSpec:
    """Create a daemon node that logs all channels to disk.

    Args:
        log_dir: Directory to write log files.
        nodes: Optional list of nodes to extract codecs from. If None, no channels
              will be logged (useful if you want to add nodes later).
        channel_filter: If provided, only log these channels. If None, log all.

    Returns:
        A NodeSpec (daemon) that can be passed to run().

    Example:
        >>> # Extract codecs from nodes automatically
        >>> logger = create_logging_node(Path("logs"), [producer, consumer])
        >>> await run([producer, consumer, logger])

        >>> # Filter specific channels
        >>> logger = create_logging_node(Path("logs"), [producer], channel_filter={"important"})
        >>> await run([producer, consumer, logger])

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

    async def logging_node(channels: dict[str, In]) -> None:
        """Log all subscribed channels to disk."""
        async with ObLogWriter(log_dir) as oblog:
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

    return NodeSpec(
        node_fn=logging_node,
        inputs={},
        outputs={},
        all_channels_param="channels",
        daemon=True,
    )


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
        >>> await run([producer, create_logging_node(Path("logs"), codecs)])
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
            elif speed > 0:
                # Speed-controlled playback with real delays
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
    )

    # Return playback node + original nodes, and first timestamp
    return [playback_spec, *nodes], first_timestamp


async def run(
    nodes: list[Callable[..., Awaitable[None]] | NodeSpec],
    log_dir: Path | None = None,
) -> None:
    """Run a graph of nodes, optionally with logging.

    This is a convenience wrapper around runtime.run_nodes() that optionally
    adds a logging node to record all output channels.

    Args:
        nodes: List of async callables (functions, bound methods, or NodeSpec objects).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name", codec_instance]
               - All channels: dict[str, In] (for monitoring/logging nodes)
        log_dir: Optional directory to log all output channels. If provided,
                automatically creates and adds a logging node.

    Raises:
        TypeError: If nodes are not async callables.
        ValueError: If channel configuration is invalid (e.g., missing input producers).

    Example:
        >>> # Simple pipeline
        >>> await run([producer, consumer])

        >>> # With automatic logging
        >>> await run([producer, consumer], log_dir=Path("logs"))

        >>> # Manual logging (more control)
        >>> logger = create_logging_node(Path("logs"), codecs, channel_filter={"important"})
        >>> await run([producer, consumer, logger])
    """
    if log_dir is not None:
        # Create logging node from nodes
        logger = create_logging_node(log_dir, nodes)
        nodes = nodes + [logger]
    await run_nodes(nodes)

async def playback(
    nodes: list[Callable[..., Awaitable[None]] | NodeSpec],
    playback_dir: Path,
    speed: float = float('inf'),
    log_dir: Path | None = None,
    use_virtual_time_logs: bool = False,
) -> None:
    """Run nodes with playback from logs, optionally logging outputs.

    This convenience function automatically creates a playback graph for any
    missing input channels and executes it with appropriate timing.

    For fast-forward mode (speed=inf), a VirtualClock is created and used
    to coordinate time between the playback node and any nodes that use
    Timer. This ensures deterministic ordering: timers scheduled for time T
    fire before messages at time T are delivered.

    Args:
        nodes: List of async callables that consume channels.
        playback_dir: Directory containing log files to play back from.
        speed: Playback speed multiplier:
               - float('inf') (default): Fast-forward (deterministic, ASAP)
               - 1.0: Realtime (respects original timestamps)
               - 2.0: Double speed
               - 0.5: Half speed
        log_dir: Optional directory to log output channels. Useful for
                recording transformed/processed data.
        use_virtual_time_logs: If True, Python log records will use the
                playback timer for timestamps instead of wall clock time.

    Raises:
        FileNotFoundError: If required log files don't exist.
        ValueError: If channel configuration is invalid.

    Example:
        >>> # Test new logic with recorded data (fast-forward)
        >>> await playback([new_planner], Path("logs"))

        >>> # Real-time playback
        >>> await playback([new_planner], Path("logs"), speed=1.0)

        >>> # Slow motion playback for debugging
        >>> await playback([new_planner], Path("logs"), speed=0.5)
        
        >>> # Playback and log transformed outputs
        >>> await playback([transform], Path("logs"), log_dir=Path("processed"))
    """
    from .logging import install_timer_log_factory, uninstall_timer_log_factory
    
    # First, create the playback graph to get the first timestamp
    # We need this before creating the timer
    clock: VirtualClock | None = None
    if speed == float('inf'):
        clock = VirtualClock()
    
    graph, first_timestamp = await create_playback_graph(nodes, playback_dir, speed=speed, clock=clock)
    
    # Create timer based on speed, initialized to playback start time
    timer: Timer
    if speed == float('inf'):
        # Fast-forward mode: use virtual clock (already created above)
        assert clock is not None
        if first_timestamp is not None:
            clock._time = first_timestamp
        timer = FastForwardTimer(clock)
    else:
        # Scaled playback: initialize timer to first message timestamp
        timer = ScaledTimer(speed, start_time=first_timestamp)

    # Install timer log factory if requested
    if use_virtual_time_logs:
        install_timer_log_factory(timer)

    try:
        if log_dir is not None:
            # Only log outputs from the user's nodes, not the playback node
            graph.append(create_logging_node(log_dir, nodes))

        await run_nodes(graph, timer=timer)
    finally:
        if use_virtual_time_logs:
            uninstall_timer_log_factory()
