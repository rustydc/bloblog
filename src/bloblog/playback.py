"""Log playback support for bloblog nodes."""
from __future__ import annotations

from collections.abc import Callable, Awaitable
from pathlib import Path
from typing import Annotated
import inspect
import struct

from .codecs import Codec, safe_unpickle_codec
from .runner import run, _parse_node_signature
from .pubsub import Out
from .bloblog import BlobLogReader

# Same header struct as bloblog.py
HEADER_STRUCT = struct.Struct('<QQ')


def read_codec_from_log(path: Path) -> Codec:
    """Read codec metadata from the first record of a bloblog file.
    
    Args:
        path: Path to the .bloblog file.
    
    Returns:
        The codec instance.
    
    Raises:
        ValueError: If codec cannot be read or unpickled.
        FileNotFoundError: If log file doesn't exist.
    """
    with open(path, 'rb') as f:
        # Read first record header
        header_bytes = f.read(16)
        if len(header_bytes) < 16:
            raise ValueError(f"Log file {path} is too short to contain codec metadata")
        
        time, data_len = HEADER_STRUCT.unpack(header_bytes)
        
        # Read codec metadata (pickled codec)
        codec_bytes = f.read(data_len)
        if len(codec_bytes) < data_len:
            raise ValueError(f"Incomplete codec metadata in {path}")
        
        # Safely unpickle the codec
        return safe_unpickle_codec(bytes(codec_bytes))


def make_log_player(
    channel_name: str,
    log_dir: Path,
    speed: float = 0,
) -> Callable[..., Awaitable[None]]:
    """Create a log player node function for a channel.
    
    Reads the codec from the log file automatically.
    
    Args:
        channel_name: Name of the channel to play back.
        log_dir: Directory containing the log files.
        speed: Playback speed multiplier (0 = fast as possible, 1.0 = realtime).
    
    Returns:
        An async node function that plays back the channel from logs.
    
    Raises:
        FileNotFoundError: If the log file doesn't exist.
        ValueError: If codec cannot be read from log.
    """
    path = log_dir / f"{channel_name}.bloblog"
    if not path.exists():
        raise FileNotFoundError(f"No log file found for channel '{channel_name}': {path}")
    
    # Read codec from log file
    codec = read_codec_from_log(path)
    
    async def log_player_node(out: Out) -> None:
        """Play back logged data for a channel."""
        # Use BlobLogReader to handle playback with speed control
        reader = BlobLogReader()
        
        first_record = True
        async def publish_callback(time: int, data: memoryview) -> None:
            nonlocal first_record
            # Skip first record (codec metadata)
            if first_record:
                first_record = False
                return
            
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
) -> list[Callable]:
    """Create log players for any inputs not produced by the live nodes.
    
    Examines the live nodes' inputs and outputs. For any input channel that
    isn't produced by a live node, creates a log player to play it back from logs.
    Codecs are automatically read from the log files.
    
    Args:
        live_nodes: List of node callables to run live.
        playback_dir: Directory containing log files for playback channels.
        playback_speed: Speed multiplier for playback (0 = fast as possible,
                        1.0 = realtime).
    
    Returns:
        A list containing the log player node functions needed to run the live nodes.
    
    Raises:
        FileNotFoundError: If any required log file doesn't exist.
        ValueError: If codec cannot be read from log files.
    """
    # Find all outputs produced by live nodes
    live_outputs: set[str] = set()
    for node_fn in live_nodes:
        _, outputs = _parse_node_signature(node_fn)
        for param_name, (channel_name, codec) in outputs.items():
            live_outputs.add(channel_name)
    
    # Find all inputs needed by live nodes
    needed_inputs: set[str] = set()
    for node_fn in live_nodes:
        inputs, _ = _parse_node_signature(node_fn)
        for param_name, channel_name in inputs.items():
            if channel_name not in live_outputs:
                needed_inputs.add(channel_name)
    
    # Create log players for missing inputs
    log_players: list[Callable] = []
    for channel_name in needed_inputs:
        log_path = playback_dir / f"{channel_name}.bloblog"
        if not log_path.exists():
            raise FileNotFoundError(f"No log file for channel '{channel_name}': {log_path}")
        log_players.append(make_log_player(channel_name, playback_dir, playback_speed))
    
    return log_players


async def playback(
    live_nodes: list[Callable],
    playback_dir: Path,
    log_dir: Path | None = None,
    speed: float = 0,
) -> None:
    """Add playback nodes for any inputs not provided by live nodes, then run all nodes.

    This is a convenience wrapper that combines make_playback_nodes() and run().
    Codecs are automatically read from log files.

    Args:
        live_nodes: The node callables to run live.
        playback_dir: Directory containing .bloblog files to play back from.
        log_dir: Optional directory to write output logs to.
        speed: Playback speed multiplier. 0 = as fast as possible, 1.0 = real-time.
    """
    playback = make_playback_nodes(live_nodes, playback_dir, speed)
    all_nodes = live_nodes + playback
    await run(all_nodes, log_dir=log_dir)

