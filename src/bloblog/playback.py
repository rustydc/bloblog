from __future__ import annotations

from pathlib import Path

from .node import Node, Input, Output, Codec, run_nodes
from .pubsub import Pub


class LogPlayer(Node):
    """A node that plays back logged channel data."""
    
    def __init__(self, channel_name: str, codec: Codec, log_dir: Path, speed: float = 0):
        self._channel_name = channel_name
        self._codec = codec
        self._log_dir = log_dir
        self._speed = speed
        # Dynamically create the output descriptor
        self._output = Output(channel_name, codec)
        self._output._attr_name = "_output_pub"
        self._output_pub: Pub | None = None  # Will be set by run_nodes
    
    def get_inputs(self) -> list[tuple[str, Input]]:
        return []
    
    def get_outputs(self) -> list[tuple[str, Output]]:
        return [("_output_pub", self._output)]
    
    async def run(self) -> None:
        from .bloblog import BlobLogReader
        
        path = self._log_dir / f"{self._channel_name}.bloblog"
        if not path.exists():
            raise FileNotFoundError(f"No log file found for channel '{self._channel_name}': {path}")
        
        assert self._output_pub is not None
        pub = self._output_pub  # Capture for closure
        
        # Use BlobLogReader to handle playback with speed control
        reader = BlobLogReader()
        
        async def publish_callback(time: int, data: bytes) -> None:
            item = self._codec.decode(data)
            await pub.publish(item)
        
        reader.handle(path, publish_callback)
        await reader.process(speed=self._speed)


def make_playback_nodes(
    live_nodes: list[Node],
    playback_dir: Path,
    playback_speed: float = 0,
) -> list[Node]:
    """Create log players for any inputs not produced by the live nodes.
    
    Examines the live nodes' inputs and outputs. For any input channel that
    isn't produced by a live node, creates a LogPlayer to play it back from logs.
    
    Args:
        live_nodes: List of nodes to run live.
        playback_dir: Directory containing log files for playback channels.
        playback_speed: Speed multiplier for playback (0 = fast as possible,
                        1.0 = realtime).
    
    Returns:
        A list containing the LogPlayers needed to run the live nodes.
    
    Raises:
        FileNotFoundError: If any required log file doesn't exist.
    """
    # Find all outputs produced by live nodes
    live_outputs: dict[str, Output] = {}
    for node in live_nodes:
        for _, out in node.get_outputs():
            live_outputs[out.name] = out
    
    # Find all inputs needed by live nodes
    needed_inputs: dict[str, Input] = {}
    for node in live_nodes:
        for _, inp in node.get_inputs():
            if inp.name not in live_outputs:
                needed_inputs[inp.name] = inp
    
    # Create log players for missing inputs
    log_players: list[Node] = []
    for name, inp in needed_inputs.items():
        log_path = playback_dir / f"{name}.bloblog"
        if not log_path.exists():
            raise FileNotFoundError(f"No log file for channel '{name}': {log_path}")
        log_players.append(LogPlayer(name, inp.codec, playback_dir, playback_speed))
    
    return log_players


async def playback_nodes(
    live_nodes: list[Node],
    playback_dir: Path,
    log_dir: Path | None = None,
    playback_speed: float = 0,
) -> None:
    """Add playback nodes for any inputs not provided by live nodes, then run all nodes.

    This is a convenience wrapper that combines add_playback_nodes() and run_nodes().

    Args:
        live_nodes: The nodes to run live.
        playback_dir: Directory containing .bloblog files to play back from.
        log_dir: Optional directory to write output logs to.
        playback_speed: Playback speed multiplier. 0 = as fast as possible, 1.0 = real-time.
    """
    nodes = live_nodes + make_playback_nodes(live_nodes, playback_dir, playback_speed)
    await run_nodes(nodes, log_dir)
