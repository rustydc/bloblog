"""Log playback support for tinman nodes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import time_ns

from .bloblog import amerge
from .oblog import read_channel_decoded
from .pubsub import Out


async def _playback_task(
    channels: dict[str, tuple[Path, Out]],
    speed: float = 0,
) -> None:
    """Read from multiple log files, merge by timestamp, decode and publish.

    Args:
        channels: Map of channel_name -> (log_path, output_channel)
        speed: Playback speed multiplier. 0 for as-fast-as-possible,
               1.0 for realtime, 2.0 for double speed, 0.5 for half speed, etc.
    """
    if not channels:
        return

    # Build mapping from index to channel info
    channel_list = list(channels.items())
    paths = [path for _name, (path, _out) in channel_list]
    
    # Create decoded readers for each channel (handles codec extraction internally)
    readers = [read_channel_decoded(path) for path in paths]

    # Track timing for speed control
    first_log_time: int | None = None
    start_wall_time: int | None = None

    try:
        # Merge the decoded records from all channels
        async for idx, time, item in amerge(*readers):
            channel_name, (_path, out) = channel_list[idx]
            
            # Apply speed control
            if speed:
                if first_log_time is None:
                    first_log_time = time
                    start_wall_time = time_ns()
                else:
                    assert start_wall_time is not None
                    log_delta = time - first_log_time
                    wall_delta = time_ns() - start_wall_time
                    wait_ns = (log_delta / speed) - wall_delta
                    if wait_ns > 0:
                        await asyncio.sleep(wait_ns / 1_000_000_000)

            # Publish already-decoded item
            await out.publish(item)
    finally:
        # Close all output channels when playback is done
        for _name, (_path, out) in channels.items():
            await out.close()
