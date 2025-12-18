"""Channel statistics tracking for tinman.

This module provides a stats node that monitors all channels and reports
statistics including message counts and throughput (Hz).

Usage:
    # Option 1: Add to node graph via CLI
    $ tinman run myapp:nodes --stats
    $ tinman playback myapp:consumer --from logs/ --stats
    
    # Option 2: Standalone stats command
    $ tinman stats --from logs/
    
    # Option 3: Programmatic use
    from tinman.stats import create_stats_node
    stats_node = create_stats_node(daemon=True)
    await run([producer, consumer, stats_node])
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from .pubsub import In
from .runtime import NodeSpec
from .timer import Timer


@dataclass
class ChannelStats:
    """Statistics for a single channel."""
    
    name: str
    count: int = 0
    first_time_ns: int | None = None
    last_time_ns: int | None = None
    
    def record_message(self, time_ns: int) -> None:
        """Record a message arrival."""
        if self.first_time_ns is None:
            self.first_time_ns = time_ns
        self.last_time_ns = time_ns
        self.count += 1
    
    @property
    def duration_seconds(self) -> float:
        """Duration from first to last message in seconds."""
        if self.first_time_ns is None or self.last_time_ns is None:
            return 0.0
        return (self.last_time_ns - self.first_time_ns) / 1e9
    
    @property
    def hz(self) -> float:
        """Average message rate in Hz."""
        duration = self.duration_seconds
        if duration <= 0 or self.count < 2:
            return 0.0
        # Use count - 1 because Hz measures intervals, not messages
        return (self.count - 1) / duration


@dataclass
class StatsCollector:
    """Collects and formats channel statistics."""
    
    channels: dict[str, ChannelStats] = field(default_factory=dict)
    
    def record(self, channel: str, time_ns: int) -> None:
        """Record a message on a channel."""
        if channel not in self.channels:
            self.channels[channel] = ChannelStats(name=channel)
        self.channels[channel].record_message(time_ns)
    
    def format_stats(self, include_header: bool = True) -> str:
        """Format statistics as a table string."""
        if not self.channels:
            return "No channels recorded."
        
        # Sort channels by name
        sorted_channels = sorted(self.channels.values(), key=lambda c: c.name)
        
        # Calculate column widths
        name_width = max(len("Channel"), max(len(c.name) for c in sorted_channels))
        count_width = max(len("Count"), max(len(str(c.count)) for c in sorted_channels))
        
        lines = []
        if include_header:
            header = f"{'Channel':<{name_width}}  {'Count':>{count_width}}  {'Hz':>10}"
            lines.append(header)
            lines.append("-" * len(header))
        
        for stats in sorted_channels:
            hz_str = f"{stats.hz:.2f}" if stats.hz > 0 else "-"
            lines.append(f"{stats.name:<{name_width}}  {stats.count:>{count_width}}  {hz_str:>10}")
        
        return "\n".join(lines)
    
    def print_stats(self, file: TextIO = sys.stdout) -> None:
        """Print formatted statistics to a file."""
        print(self.format_stats(), file=file)


def create_stats_node(
    daemon: bool = True,
    print_interval: float | None = None,
    print_on_complete: bool = True,
    output: TextIO = sys.stderr,
) -> NodeSpec:
    """Create a node that collects and reports channel statistics.
    
    Args:
        daemon: If True, node is cancelled when main nodes complete (default).
                Set to False for standalone stats mode.
        print_interval: If set, print stats every N seconds (useful for live monitoring).
        print_on_complete: If True, print final stats when node completes.
        output: File to print stats to (default: stderr).
    
    Returns:
        A NodeSpec that can be added to a node graph.
    
    Example:
        >>> # As a daemon (won't block shutdown)
        >>> stats = create_stats_node()
        >>> await run([producer, consumer, stats])
        
        >>> # With periodic updates
        >>> stats = create_stats_node(print_interval=1.0)
        >>> await run([producer, consumer, stats])
        
        >>> # Standalone mode (blocks until all channels close)
        >>> stats = create_stats_node(daemon=False)
        >>> await playback([], playback_dir=Path("logs"))
    """
    collector = StatsCollector()
    
    async def stats_node(channels: dict[str, In], timer: Timer) -> None:
        """Collect statistics from all channels."""
        
        async def collect_channel(name: str, channel: In) -> None:
            """Collect stats for a single channel."""
            async for _item in channel:
                collector.record(name, timer.time_ns())
        
        async def periodic_printer() -> None:
            """Print stats periodically if configured."""
            if print_interval is None:
                return
            while True:
                await timer.sleep(print_interval)
                print(f"\n--- Stats at {timer.time_ns() / 1e9:.3f}s ---", file=output)
                collector.print_stats(file=output)
        
        try:
            async with asyncio.TaskGroup() as tg:
                # Start collector for each channel
                for name, channel in channels.items():
                    tg.create_task(collect_channel(name, channel))
                
                # Start periodic printer if configured
                if print_interval is not None:
                    tg.create_task(periodic_printer())
        except* asyncio.CancelledError:
            # Expected when running as daemon
            pass
        finally:
            if print_on_complete and collector.channels:
                print("\n--- Final Channel Stats ---", file=output)
                collector.print_stats(file=output)
    
    return NodeSpec(
        node_fn=stats_node,
        inputs={},
        outputs={},
        all_channels_param="channels",
        timer_param="timer",
        daemon=daemon,
    )


async def run_stats(
    log_dir: Path,
    output: TextIO = sys.stdout,
) -> None:
    """Run standalone stats analysis on a log directory.
    
    This discovers all channels in the log directory and prints statistics
    for each one. Unlike using a stats node in playback, this reads directly
    from the log files without needing to set up a full playback graph.
    
    Args:
        log_dir: Directory containing .blog log files.
        output: File to print stats to (default: stdout).
    
    Example:
        >>> await run_stats(Path("webcam_logs"))
        Channel  Count        Hz
        -------------------------
        camera     100     30.00
    """
    from .oblog import ObLog
    
    log_dir = Path(log_dir)
    
    # Discover all channels (*.blog files)
    blog_files = list(log_dir.glob("*.blog"))
    if not blog_files:
        print(f"No log files found in {log_dir}", file=output)
        return
    
    channels = [f.stem for f in blog_files]
    collector = StatsCollector()
    
    # Pre-register all channels so empty ones show up with count=0
    for channel in channels:
        collector.channels[channel] = ChannelStats(name=channel)
    
    async def collect_channel(channel: str) -> None:
        """Collect stats for a single channel."""
        oblog = ObLog(log_dir)
        try:
            async for timestamp, _item in oblog.read_channel(channel):
                collector.record(channel, timestamp)
        finally:
            await oblog.close()
    
    async with asyncio.TaskGroup() as tg:
        for channel in channels:
            tg.create_task(collect_channel(channel))
    
    collector.print_stats(file=output)
