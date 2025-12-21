"""Graph building and execution for tinman pipelines.

This module provides the Graph class for composing and running node pipelines.
Transform functions for specific features (recording, playback, logging, stats)
are in their respective modules.

Example:
    >>> from tinman import Graph
    >>> from tinman.recording import with_recording
    >>> from tinman.playback import with_playback
    >>> from tinman.stats import with_stats
    >>> 
    >>> # Build and run a pipeline
    >>> graph = Graph.of(producer, transformer, consumer)
    >>> graph = with_recording(Path("logs"))(graph)
    >>> graph = with_stats()(graph)
    >>> await graph.run()
    >>> 
    >>> # Playback recorded data
    >>> graph = Graph.of(new_consumer)
    >>> graph = (await with_playback(Path("logs")))(graph)
    >>> await graph.run()
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .graphviz import generate_dot
from .runtime import NodeSpec, get_node_specs, run_nodes
from .timer import Timer

if TYPE_CHECKING:
    from .logging import LogHandler
    from .timer import VirtualClock


@dataclass
class Graph:
    """A node graph with metadata for visualization and execution.
    
    The Graph tracks which nodes are "user nodes" (provided by the user)
    vs "system nodes" (added by transforms like with_recording). This
    distinction is used when generating DOT visualizations.
    
    Attributes:
        nodes: All nodes in the graph (user + system).
        user_node_names: Names of user-provided nodes.
        timer: Timer for playback modes (None for live execution).
        _log_handler: Internal handler for log capture cleanup.
        _playback_channels: PlaybackIn channels for playback mode.
        _clock: VirtualClock for fast-forward playback.
    """
    nodes: list[Callable[..., Awaitable[None]] | NodeSpec]
    user_node_names: set[str]
    timer: Timer | None = None
    _log_handler: "LogHandler | None" = field(default=None, repr=False)
    _playback_channels: dict = field(default_factory=dict, repr=False)
    _clock: "VirtualClock | None" = field(default=None, repr=False)
    
    @classmethod
    def of(cls, *nodes: Callable[..., Awaitable[None]] | NodeSpec) -> Graph:
        """Create a graph from user nodes.
        
        Args:
            *nodes: Node functions or NodeSpec objects.
            
        Returns:
            A new Graph containing the given nodes.
            
        Example:
            >>> graph = Graph.of(producer, transformer, consumer)
        """
        names: set[str] = set()
        for n in nodes:
            if isinstance(n, NodeSpec):
                if n.name is not None:
                    names.add(n.name)
            else:
                names.add(n.__name__)
        return cls(list(nodes), names)
    
    def to_dot(self) -> str:
        """Generate Graphviz DOT representation.
        
        Returns:
            DOT format string that can be rendered with Graphviz.
            
        Example:
            >>> print(graph.to_dot())
            >>> # Or save to file
            >>> Path("graph.dot").write_text(graph.to_dot())
        """
        specs = get_node_specs(self.nodes)
        return generate_dot(specs, self.user_node_names)
    
    def write_dot(self, path: Path | str) -> None:
        """Write DOT representation to a file.
        
        Args:
            path: Path to write the DOT file.
            
        Example:
            >>> graph.write_dot("pipeline.dot")
        """
        Path(path).write_text(self.to_dot())
    
    async def run(self) -> None:
        """Execute the graph.
        
        Runs all nodes concurrently until completion. If a timer was
        set (e.g., by with_playback), it's passed to the runtime.
        
        For fast-forward playback, a clock driver task processes events
        from the VirtualClock's priority queue.
        
        Any attached log handlers are cleaned up after execution.
        
        Example:
            >>> await graph.run()
        """
        try:
            await run_nodes(
                self.nodes, 
                timer=self.timer,
                playback_channels=self._playback_channels,
                clock=self._clock,
            )
        finally:
            # Clean up log handler if attached
            if self._log_handler is not None:
                self._log_handler.close()
