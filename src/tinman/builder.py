"""Graph building with composable transforms.

This module provides a functional approach to building node graphs with
optional system nodes like logging, stats, and playback.

Example:
    >>> graph = Graph.of(producer, transformer, consumer)
    >>> 
    >>> if log_dir:
    ...     graph = with_logging(log_dir)(graph)
    >>> if stats:
    ...     graph = with_stats()(graph)
    >>> 
    >>> # Generate visualization
    >>> print(graph.to_dot())
    >>> 
    >>> # Run the graph
    >>> await graph.run()
"""

from __future__ import annotations

import logging as python_logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .graph import generate_dot
from .logging import LogHandler, create_log_printer
from .pubsub import In, Out
from .runtime import NodeSpec, get_node_specs, run_nodes
from .stats import create_stats_node
from .timer import Timer, ScaledTimer, FastForwardTimer, VirtualClock

if TYPE_CHECKING:
    pass


@dataclass
class Graph:
    """A node graph with metadata for visualization and execution.
    
    The Graph tracks which nodes are "user nodes" (provided by the user)
    vs "system nodes" (added by transforms like with_logging). This
    distinction is used when generating DOT visualizations.
    
    Attributes:
        nodes: All nodes in the graph (user + system).
        user_node_names: Names of user-provided nodes.
        timer: Timer for playback modes (None for live execution).
        _log_handler: Internal handler for log capture cleanup.
    """
    nodes: list[Callable[..., Awaitable[None]] | NodeSpec]
    user_node_names: set[str]
    timer: Timer | None = None
    _log_handler: LogHandler | None = field(default=None, repr=False)
    
    @classmethod
    def of(cls, *nodes: Callable[..., Awaitable[None]] | NodeSpec) -> "Graph":
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
    
    async def run(self) -> None:
        """Execute the graph.
        
        Runs all nodes concurrently until completion. If a timer was
        set (e.g., by with_playback), it's passed to the runtime.
        
        Any attached log handlers are cleaned up after execution.
        """
        try:
            await run_nodes(self.nodes, timer=self.timer)
        finally:
            # Clean up log handler if attached
            if self._log_handler is not None:
                self._log_handler.close()


# =============================================================================
# Transform functions
# =============================================================================

def with_logging(log_dir: Path) -> Callable[[Graph], Graph]:
    """Add a logging node that records all channels to disk.
    
    Args:
        log_dir: Directory to write log files.
        
    Returns:
        A transform function that adds logging to a graph.
        
    Example:
        >>> graph = with_logging(Path("logs"))(graph)
    """
    def transform(g: Graph) -> Graph:
        from .launcher import create_logging_node
        g.nodes.append(create_logging_node(log_dir, g.nodes))
        return g
    return transform


def with_stats(
    print_interval: float | None = None,
    print_on_complete: bool = True,
) -> Callable[[Graph], Graph]:
    """Add a stats node that reports channel throughput.
    
    Args:
        print_interval: If set, print stats every N seconds.
        print_on_complete: If True, print final stats when done.
        
    Returns:
        A transform function that adds stats to a graph.
        
    Example:
        >>> graph = with_stats()(graph)
        >>> graph = with_stats(print_interval=1.0)(graph)
    """
    def transform(g: Graph) -> Graph:
        g.nodes.append(create_stats_node(
            daemon=True,
            print_interval=print_interval,
            print_on_complete=print_on_complete,
        ))
        return g
    return transform


def with_log_capture(
    channel: str = "logs",
    level: int = python_logging.INFO,
    logger: python_logging.Logger | str | None = None,
    print_logs: bool = True,
) -> Callable[[Graph], Graph]:
    """Capture Python logging to a channel.
    
    Attaches a handler to the Python logging system that captures log
    records and publishes them to a tinman channel. The handler is
    automatically cleaned up when the graph finishes running.
    
    Args:
        channel: Output channel name for log entries.
        level: Minimum log level to capture.
        logger: Logger to attach to. None = root logger, str = logger name.
        print_logs: If True, also add a log printer node.
        
    Returns:
        A transform function that adds log capture to a graph.
        
    Example:
        >>> graph = with_log_capture()(graph)
        >>> graph = with_log_capture(level=logging.DEBUG, logger="myapp")(graph)
    """
    def transform(g: Graph) -> Graph:
        handler = LogHandler(channel=channel, level=level)
        
        # Attach to logger
        if logger is None:
            target_logger = python_logging.getLogger()
        elif isinstance(logger, str):
            target_logger = python_logging.getLogger(logger)
        else:
            target_logger = logger
        target_logger.addHandler(handler)
        
        # Add capture node
        g.nodes.append(handler.node)
        
        # Add printer node if requested
        if print_logs:
            g.nodes.append(create_log_printer(channel))
        
        # Store handler for cleanup
        g._log_handler = handler
        
        return g
    return transform


async def with_playback(
    playback_dir: Path,
    speed: float = float('inf'),
    use_virtual_time_logs: bool = False,
) -> Callable[[Graph], Graph]:
    """Inject playback nodes for missing inputs.
    
    Analyzes the graph to find which channels are needed but not produced,
    then creates playback nodes to provide those channels from recorded logs.
    
    This is an async function because it reads log file metadata.
    
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
        >>> graph = (await with_playback(Path("logs")))(graph)
        >>> graph = (await with_playback(Path("logs"), speed=1.0))(graph)
    """
    from .launcher import create_playback_graph
    
    # We need to defer the actual playback graph creation until we have the graph's nodes.
    # But create_playback_graph is async and the transform should be sync.
    # Solution: capture parameters and do the work in transform using run_until_complete.
    
    def transform(g: Graph) -> Graph:
        import asyncio
        
        # Create clock for fast-forward mode
        clock: VirtualClock | None = None
        if speed == float('inf'):
            clock = VirtualClock()
        
        # Run async playback graph creation
        loop = asyncio.get_event_loop()
        playback_nodes, first_timestamp = loop.run_until_complete(
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
