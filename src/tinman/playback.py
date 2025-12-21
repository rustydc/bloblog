"""Playback transforms for replaying recorded data.

This module provides transforms for playing back recorded channel data
from ObLog files, with support for various playback speeds.

Key functions:
- create_playback_graph: Create a Graph configured for playback
- with_playback: Transform to inject playback into an existing graph

Example:
    >>> from tinman import Graph
    >>> from tinman.playback import with_playback
    >>> 
    >>> graph = Graph.of(consumer)
    >>> graph = with_playback(Path("logs"))(graph)
    >>> await graph.run()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING

from .oblog import Codec, ObLogReader
from .pubsub import In
from .runtime import NodeSpec, get_node_specs
from .timer import ScaledTimer, FastForwardTimer, VirtualClock

if TYPE_CHECKING:
    from .launcher import Graph


class PlaybackIn[T]:
    """Input channel that reads from recorded logs with event-driven timing.

    Each PlaybackIn reads from a blog file and schedules events in the
    VirtualClock. When __anext__ is called, it reads the next message
    from the log, schedules an event at that message's timestamp, and
    waits for the clock to process it.

    Multiple PlaybackIn instances (for different channels or different
    subscribers to the same channel) naturally interleave through the
    shared clock's priority queue.

    For speed-controlled playback (not fast-forward), timing is handled
    differently: we use wall-clock delays based on recorded timestamps.
    
    Supports time interval filtering via start_ts and end_ts parameters.
    """

    def __init__(
        self,
        reader: AsyncGenerator[tuple[int, T], None],
        clock: VirtualClock | None = None,
        speed: float = float('inf'),
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> None:
        """Initialize a playback input channel.

        Args:
            reader: Async generator yielding (timestamp, item) from ObLogReader.
            clock: VirtualClock for fast-forward mode (required if speed=inf).
            speed: Playback speed multiplier (inf = fast-forward).
            start_ts: Skip records before this timestamp (nanoseconds). None means
                      start from the beginning.
            end_ts: Stop after this timestamp (nanoseconds). None means play to end.
        """
        self._reader = reader
        self._clock = clock
        self._speed = speed
        self._start_ts = start_ts
        self._end_ts = end_ts
        self._timestamp: int = 0
        self._start_time: int | None = None
        self._start_real: int | None = None
        self._exhausted = False

    def __aiter__(self) -> "PlaybackIn[T]":
        return self

    async def __anext__(self) -> T:
        if self._exhausted:
            raise StopAsyncIteration

        while True:
            try:
                timestamp, item = await anext(self._reader)
            except StopAsyncIteration:
                self._exhausted = True
                raise

            # Check end boundary
            if self._end_ts is not None and timestamp > self._end_ts:
                self._exhausted = True
                raise StopAsyncIteration

            # Skip records before start boundary
            if self._start_ts is not None and timestamp < self._start_ts:
                continue

            break

        # Handle timing based on mode
        if self._speed == float('inf'):
            # Fast-forward: schedule event in clock and wait
            assert self._clock is not None
            await self._clock.schedule(timestamp)
        elif self._speed > 0:
            # Speed-controlled: delay based on wall clock
            if self._start_time is None:
                self._start_time = timestamp
                self._start_real = time_ns()
            else:
                elapsed_log = timestamp - self._start_time
                assert self._start_real is not None
                target_real = self._start_real + int(elapsed_log / self._speed)
                current = time_ns()
                if current < target_real:
                    await asyncio.sleep((target_real - current) / 1e9)

        self._timestamp = timestamp
        return item

    @property
    def timestamp(self) -> int:
        """Timestamp of the most recently read message."""
        return self._timestamp


async def create_playback_graph(
    graph: "Graph",
    playback_dir: Path,
    speed: float = float('inf'),
    start_offset: float | None = None,
    end_offset: float | None = None,
) -> "Graph":
    """Configure a graph for playback from recorded logs.

    This function analyzes the graph to find which channels are needed,
    checks which channels are missing (not produced by any node), and creates
    PlaybackIn instances to provide those channels from recorded logs.

    Args:
        graph: The graph to configure for playback.
        playback_dir: Directory containing .blog log files.
        speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed, 
               float('inf') = fast-forward as fast as possible).
        start_offset: Start playback at this offset (in seconds) from the
                      first timestamp in the log. None means start from beginning.
        end_offset: End playback at this offset (in seconds) from the first
                    timestamp in the log. None means play to the end.

    Returns:
        The configured graph (same instance, mutated in place).

    Raises:
        FileNotFoundError: If a required log file doesn't exist.

    Example:
        >>> graph = await create_playback_graph(
        ...     Graph.of(consumer),
        ...     Path("logs"),
        ...     speed=float('inf'),
        ... )
        >>> await graph.run()
        
        >>> # Play only 10s to 30s interval
        >>> graph = await create_playback_graph(
        ...     Graph.of(consumer),
        ...     Path("logs"),
        ...     start_offset=10.0,
        ...     end_offset=30.0,
        ... )
    """
    # Create clock for fast-forward mode
    clock: VirtualClock | None = None
    if speed == float('inf'):
        clock = VirtualClock()

    # Parse nodes to find what channels are needed and what's provided
    specs = get_node_specs(graph.nodes)
    all_inputs: set[str] = set()
    all_outputs: set[str] = set()

    # Count subscribers per channel
    channel_subscriber_count: dict[str, int] = {}
    for spec in specs:
        for _, (channel_name, _) in spec.inputs.items():
            all_inputs.add(channel_name)
            channel_subscriber_count[channel_name] = channel_subscriber_count.get(channel_name, 0) + 1
        for _, (channel_name, _) in spec.outputs.items():
            all_outputs.add(channel_name)

    # Find channels that need playback
    missing_channels = all_inputs - all_outputs

    playback_channels: dict[str, tuple[Codec, list[PlaybackIn]]] = {}
    first_timestamp: int | None = None

    if missing_channels:
        # Read codecs and first timestamps for all missing channels
        oblog = ObLogReader(playback_dir)

        for channel in missing_channels:
            try:
                codec = await oblog.read_codec(channel)
                # Get first timestamp for this channel
                ts = await oblog.first_timestamp(channel)
                if ts is not None and (first_timestamp is None or ts < first_timestamp):
                    first_timestamp = ts

                playback_channels[channel] = (codec, [])  # PlaybackIns added below

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No log file for channel '{channel}' in {playback_dir}"
                )

        # Calculate absolute start/end timestamps from offsets
        start_ts: int | None = None
        end_ts: int | None = None
        if first_timestamp is not None:
            if start_offset is not None:
                start_ts = first_timestamp + int(start_offset * 1e9)
            if end_offset is not None:
                end_ts = first_timestamp + int(end_offset * 1e9)

        # Create PlaybackIn instances with interval filtering
        for channel in missing_channels:
            codec, _ = playback_channels[channel]
            num_subscribers = channel_subscriber_count.get(channel, 1)
            playback_ins: list[PlaybackIn] = []
            for _ in range(num_subscribers):
                # Each subscriber gets its own reader (sharing the mmap)
                reader = oblog.read_channel(channel)
                playback_in = PlaybackIn(
                    reader, clock=clock, speed=speed,
                    start_ts=start_ts, end_ts=end_ts,
                )
                playback_ins.append(playback_in)
            playback_channels[channel] = (codec, playback_ins)

    # Configure the graph
    graph._playback_channels = playback_channels
    graph._clock = clock
    
    # Create timer based on speed
    effective_start = first_timestamp
    if first_timestamp is not None and start_offset is not None:
        effective_start = first_timestamp + int(start_offset * 1e9)
    
    if speed == float('inf'):
        assert clock is not None
        if effective_start is not None:
            clock._time = effective_start
        graph.timer = FastForwardTimer(clock)
    else:
        graph.timer = ScaledTimer(speed, start_time=effective_start)

    return graph


def with_playback(
    playback_dir: Path,
    speed: float = float('inf'),
    use_virtual_time_logs: bool = False,
    start_offset: float | None = None,
    end_offset: float | None = None,
) -> Callable[["Graph"], "Graph"]:
    """Inject playback for missing inputs.
    
    This is a convenience wrapper around create_playback_graph() that can be
    used synchronously before entering an async context.
    
    Analyzes the graph to find which channels are needed but not produced,
    then creates PlaybackIn channels to provide those from recorded logs.
    
    Note: This uses asyncio.run() internally, so it cannot be called from
    within an existing async event loop. If you're already in an async
    context, use create_playback_graph() directly.
    
    Args:
        playback_dir: Directory containing .blog log files.
        speed: Playback speed multiplier:
               - float('inf') (default): Fast-forward (deterministic, ASAP)
               - 1.0: Realtime (respects original timestamps)
               - 2.0: Double speed
        use_virtual_time_logs: If True, Python log records use playback time.
        start_offset: Start playback at this offset (in seconds) from the 
                      beginning of the log. None means start from beginning.
        end_offset: End playback at this offset (in seconds) from the beginning
                    of the log. None means play to the end.
        
    Returns:
        A transform function that adds playback to a graph.
        
    Example:
        >>> graph = Graph.of(consumer)
        >>> graph = with_playback(Path("logs"))(graph)
        >>> await graph.run()
        
        >>> # Play only the interval from 10s to 30s
        >>> graph = with_playback(Path("logs"), start_offset=10, end_offset=30)(graph)
    """
    def transform(g: "Graph") -> "Graph":
        # Run async playback graph creation synchronously
        # This is safe because we're not yet in an event loop
        configured = asyncio.run(
            create_playback_graph(
                g, playback_dir, speed=speed,
                start_offset=start_offset, end_offset=end_offset,
            )
        )
        
        # Handle virtual time logs
        if use_virtual_time_logs and configured.timer is not None:
            from .logging import install_timer_log_factory
            install_timer_log_factory(configured.timer)
        
        return configured
    
    return transform
