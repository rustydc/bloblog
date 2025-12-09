"""Log playback support for bloblog nodes."""
from __future__ import annotations

from collections.abc import Callable, Awaitable
from pathlib import Path
from typing import Annotated
import inspect

from .node import Codec, run_nodes, _parse_node_signature
from .pubsub import Out


def make_log_player(
    channel_name: str,
    codec: Codec,
    log_dir: Path,
    speed: float = 0,
) -> Callable[..., Awaitable[None]]:
    """Create a log player node function for a channel.
    
    Args:
        channel_name: Name of the channel to play back.
        codec: Codec for decoding the channel data.
        log_dir: Directory containing the log files.
        speed: Playback speed multiplier (0 = fast as possible, 1.0 = realtime).
    
    Returns:
        An async node function that plays back the channel from logs.
    
    Raises:
        FileNotFoundError: If the log file doesn't exist.
    """
    from .bloblog import BlobLogReader
    
    path = log_dir / f"{channel_name}.bloblog"
    if not path.exists():
        raise FileNotFoundError(f"No log file found for channel '{channel_name}': {path}")
    
    async def log_player_node(out: Out) -> None:
        """Play back logged data for a channel."""
        # Use BlobLogReader to handle playback with speed control
        reader = BlobLogReader()
        
        async def publish_callback(time: int, data: memoryview) -> None:
            # Decode directly from memoryview (zero-copy)
            item = codec.decode(data)
            await out.publish(item)
        
        reader.handle(path, publish_callback)
        await reader.process(speed=speed)
    
    # Set a descriptive name for the function
    log_player_node.__name__ = f"log_player_{channel_name}"
    
    # Manually attach type annotations that _parse_node_signature will understand
    # We can't use Annotated in the actual signature because channel_name is dynamic,
    # so we store the metadata directly. We need to use Out[Any] as the base type.
    from typing import Any
    log_player_node.__annotations__ = {
        "out": Annotated[Out[Any], channel_name, codec],
        "return": None,
    }
    
    return log_player_node


def make_playback_nodes(
    live_nodes: list[Callable],
    playback_dir: Path,
    playback_speed: float = 0,
    codec_registry: dict[str, Codec] | None = None,
) -> list[Callable]:
    """Create log players for any inputs not produced by the live nodes.
    
    Examines the live nodes' inputs and outputs. For any input channel that
    isn't produced by a live node, creates a log player to play it back from logs.
    
    Args:
        live_nodes: List of node callables to run live.
        playback_dir: Directory containing log files for playback channels.
        playback_speed: Speed multiplier for playback (0 = fast as possible,
                        1.0 = realtime).
        codec_registry: Registry of channel codecs. If not provided, codecs must
                        be in node annotations.
    
    Returns:
        A list containing the log player node functions needed to run the live nodes.
    
    Raises:
        FileNotFoundError: If any required log file doesn't exist.
    """
    if codec_registry is None:
        codec_registry = {}
    
    # Find all outputs produced by live nodes
    live_outputs: dict[str, Codec] = {}
    for node_fn in live_nodes:
        _, outputs = _parse_node_signature(node_fn, codec_registry)
        for param_name, (channel_name, codec) in outputs.items():
            live_outputs[channel_name] = codec
    
    # Find all inputs needed by live nodes
    needed_inputs: dict[str, Codec] = {}
    for node_fn in live_nodes:
        inputs, _ = _parse_node_signature(node_fn, codec_registry)
        for param_name, (channel_name, codec) in inputs.items():
            if channel_name not in live_outputs:
                needed_inputs[channel_name] = codec
    
    # Create log players for missing inputs
    log_players: list[Callable] = []
    for channel_name, codec in needed_inputs.items():
        log_path = playback_dir / f"{channel_name}.bloblog"
        if not log_path.exists():
            raise FileNotFoundError(f"No log file for channel '{channel_name}': {log_path}")
        log_players.append(make_log_player(channel_name, codec, playback_dir, playback_speed))
    
    return log_players


async def playback_nodes(
    live_nodes: list[Callable],
    playback_dir: Path,
    log_dir: Path | None = None,
    playback_speed: float = 0,
    channels: dict[str, Codec] | None = None,
) -> None:
    """Add playback nodes for any inputs not provided by live nodes, then run all nodes.

    This is a convenience wrapper that combines make_playback_nodes() and run_nodes().

    Args:
        live_nodes: The node callables to run live.
        playback_dir: Directory containing .bloblog files to play back from.
        log_dir: Optional directory to write output logs to.
        playback_speed: Playback speed multiplier. 0 = as fast as possible, 1.0 = real-time.
        channels: Optional dict mapping channel names to codecs (same as run_nodes).
    """
    # Build codec registry from channels if provided
    codec_registry: dict[str, Codec] = channels if channels else {}
    
    playback = make_playback_nodes(live_nodes, playback_dir, playback_speed, codec_registry)
    all_nodes = live_nodes + playback
    await run_nodes(all_nodes, channels=channels, log_dir=log_dir)
