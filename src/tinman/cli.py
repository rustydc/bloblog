"""Tinman command-line interface.

Usage:
    tinman run module:node1,node2 other_module:node3 --log-dir logs/
    tinman playback module:consumer --from logs/ --speed 2.0
    tinman playback module:consumer --from logs/  # fast-forward (default)
    tinman run myapp:nodes --log-dir logs/ --capture-logs  # capture Python logs
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

import cyclopts

from .launcher import Graph
from .playback import with_playback
from .recording import create_recording_node
from .runtime import NodeSpec, run_nodes
from .oblog import enable_pickle_codec
from .logging import log_capture_context, create_log_printer
from .stats import create_stats_node

app = cyclopts.App(
    name="tinman",
    help="A minimal async framework for data pipelines with logging and playback.",
)


# Default log directory management
DEFAULT_LOG_BASE = Path.home() / ".tinman" / "logs"
MAX_LOG_DIRS = 5


def _get_default_log_dir() -> Path:
    """Create a new timestamped log directory under ~/.tinman/logs/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = DEFAULT_LOG_BASE / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _cleanup_old_log_dirs() -> None:
    """Remove old log directories, keeping only the most recent MAX_LOG_DIRS - 1.
    
    This is called before creating a new directory, so we keep 4 old ones
    plus the new one = 5 total.
    """
    if not DEFAULT_LOG_BASE.exists():
        return
    
    # Get all directories sorted by name (which is timestamp, so oldest first)
    dirs = sorted(
        [d for d in DEFAULT_LOG_BASE.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )
    
    # Remove all but the most recent (MAX_LOG_DIRS - 1)
    dirs_to_remove = dirs[: -(MAX_LOG_DIRS - 1)] if len(dirs) >= MAX_LOG_DIRS else []
    for d in dirs_to_remove:
        shutil.rmtree(d)


def _get_latest_log_dir() -> Path | None:
    """Get the most recent log directory under ~/.tinman/logs/."""
    if not DEFAULT_LOG_BASE.exists():
        return None
    
    dirs = sorted(
        [d for d in DEFAULT_LOG_BASE.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True
    )
    return dirs[0] if dirs else None


def _resolve_attr(obj: object, name: str) -> object:
    """Resolve a dotted attribute path like 'MyClass.method'."""
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _is_no_arg_callable(obj: object) -> bool:
    """Check if obj is a callable that takes no required arguments."""
    import inspect
    if not callable(obj):
        return False
    try:
        sig = inspect.signature(obj)
        # Check if all parameters have defaults or are *args/**kwargs
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return False
        return True
    except (ValueError, TypeError):
        # Can't inspect signature (e.g., built-in), assume not a factory
        return False


def _is_node(obj: object) -> bool:
    """Check if obj looks like a node (async function or has node-like signature)."""
    import inspect
    # Async functions are definitely nodes
    if inspect.iscoroutinefunction(obj):
        return True
    # Bound methods of async functions
    if inspect.ismethod(obj) and inspect.iscoroutinefunction(obj.__func__):
        return True
    # NodeSpec is a node
    if isinstance(obj, NodeSpec):
        return True
    return False


def load_nodes(specs: tuple[str, ...]) -> list[Callable | NodeSpec]:
    """Load nodes from 'module.path:name1,name2' specs.

    Args:
        specs: One or more node specifications in 'module:name1,name2' format.
               Multiple specs are combined into a single node list.

    Each name can be:
        - An async function or NodeSpec (node)
        - A list of nodes
        - A no-arg factory function that returns a node or list of nodes

    Returns:
        List of node callables ready to pass to run() or playback().

    Examples:
        >>> load_nodes(("myapp.nodes:producer,consumer",))
        [<function producer>, <function consumer>]

        >>> load_nodes(("myapp.sensors:camera", "myapp.processors:detector,tracker"))
        [<function camera>, <function detector>, <function tracker>]

        >>> load_nodes(("myapp:graph",))  # 'graph' is a list
        [<function node1>, <function node2>, ...]
        
        >>> load_nodes(("myapp:create_detector",))  # factory function
        [<function detector>]  # result of create_detector()
    """
    nodes: list[Callable] = []

    for spec in specs:
        module_path, _, names = spec.partition(":")
        if not names:
            raise ValueError(
                f"Missing node names in '{spec}', expected 'module:name1,name2'"
            )

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not import module '{module_path}': {e}"
            ) from e

        for name in names.split(","):
            name = name.strip()
            if not name:
                continue

            try:
                obj = _resolve_attr(module, name)
            except AttributeError as e:
                raise AttributeError(
                    f"Module '{module_path}' has no attribute '{name}'"
                ) from e

            # If it's already a list, extend
            if isinstance(obj, list):
                nodes.extend(obj)
            # If it's a node (async function), add it directly
            elif _is_node(obj):
                nodes.append(obj)
            # If it's a no-arg callable (factory), call it
            elif callable(obj) and _is_no_arg_callable(obj):
                result = obj()
                if isinstance(result, list):
                    nodes.extend(result)
                elif callable(result) or isinstance(result, NodeSpec):
                    nodes.append(result)
                else:
                    raise TypeError(
                        f"Factory '{module_path}:{name}' returned {type(result).__name__}, "
                        f"expected a node or list of nodes"
                    )
            elif callable(obj):
                # Callable but requires args - can't be used as factory, treat as node
                nodes.append(obj)
            else:
                raise TypeError(
                    f"'{module_path}:{name}' is not callable or a list of nodes, "
                    f"got {type(obj).__name__}"
                )

    if not nodes:
        raise ValueError("No nodes specified")

    return nodes


@app.command
def run(
    nodes: tuple[str, ...],
    *,
    log_dir: Path | None = None,
    no_log: bool = False,
    pickle: bool = False,
    capture_logs: bool = True,
    log_level: str = "INFO",
    log_channel: str = "logs",
    stats: bool = False,
    graph: Path | None = None,
) -> None:
    """Run a node graph.

    Parameters
    ----------
    nodes
        One or more node specs in 'module:node1,node2' format.
        Multiple specs are combined into a single graph.
    log_dir
        Directory to log all output channels. Default: ~/.tinman/logs/TIMESTAMP/
    no_log
        Disable logging to disk entirely.
    pickle
        Enable PickleCodec for arbitrary Python objects (security risk with untrusted data).
    capture_logs
        Capture Python logging output to a channel. Adds a handler to the root logger
        and includes a log capture node in the graph.
    log_level
        Minimum log level to capture when --capture-logs is enabled.
        One of: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default: INFO.
    log_channel
        Channel name for captured logs. Default: "logs".
    stats
        Print channel statistics (message counts and Hz) on exit.
    graph
        Output a Graphviz DOT file of the node graph.

    Examples
    --------
    $ tinman run myapp.nodes:producer,consumer
    $ tinman run myapp.sensors:camera myapp.processors:detector --log-dir logs/
    $ tinman run myapp:nodes --no-log  # disable logging
    $ tinman run myapp:nodes --capture-logs --log-level DEBUG
    $ tinman run myapp:nodes --stats
    $ tinman run myapp:nodes --graph graph.dot
    """
    if pickle:
        enable_pickle_codec()
    
    # Determine log directory
    if no_log:
        effective_log_dir = None
    elif log_dir is not None:
        effective_log_dir = log_dir
    else:
        # Default: ~/.tinman/logs/TIMESTAMP/
        _cleanup_old_log_dirs()
        effective_log_dir = _get_default_log_dir()
    
    node_list = load_nodes(nodes)
    
    # Build graph from user nodes
    g = Graph.of(*node_list)
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    # Set root logger level to allow log capture at the specified level
    if capture_logs:
        logging.getLogger().setLevel(level)
    with log_capture_context(capture_logs, channel=log_channel, level=level) as log_nodes:
        # Add log capture nodes
        g.nodes.extend(log_nodes)
        
        # Add log printer if we're capturing logs
        if capture_logs:
            g.nodes.append(create_log_printer(log_channel))
        
        # Add stats if requested
        if stats:
            g.nodes.append(create_stats_node(daemon=True))
        
        # Add recording
        if effective_log_dir is not None:
            g.nodes.append(create_recording_node(effective_log_dir, g.nodes))
        
        # If --graph specified, output DOT
        if graph is not None:
            graph.write_text(g.to_dot())
        
        asyncio.run(g.run())


@app.command
def playback(
    nodes: tuple[str, ...],
    *,
    from_: Annotated[Path | None, cyclopts.Parameter(name=["--from", "-f"])] = None,
    speed: float = float("inf"),
    log_dir: Path | None = None,
    pickle: bool = False,
    capture_logs: bool = True,
    log_level: str = "INFO",
    log_channel: str = "logs",
    use_virtual_time: bool = True,
    stats: bool = False,
    graph: Path | None = None,
) -> None:
    """Play back recorded logs through nodes.

    Parameters
    ----------
    nodes
        One or more node specs in 'module:node1,node2' format.
        These consume channels from the recorded logs.
    from_
        Directory containing log files to play back. Default: latest ~/.tinman/logs/
    speed
        Playback speed multiplier. Use 'inf' for fast-forward (default),
        1.0 for real-time, 2.0 for double speed, etc.
    log_dir
        Directory to log output channels (for recording transformed data).
    pickle
        Enable PickleCodec for arbitrary Python objects (security risk with untrusted data).
    capture_logs
        Capture Python logging output to a channel. Adds a handler to the root logger
        and includes a log capture node in the graph.
    log_level
        Minimum log level to capture when --capture-logs is enabled.
        One of: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default: INFO.
    log_channel
        Channel name for captured logs. Default: "logs".
    use_virtual_time
        Use virtual/scaled time for Python logging timestamps. When enabled,
        log messages will have timestamps matching the playback time rather
        than wall clock time. Default: True.
    stats
        Print channel statistics (message counts and Hz) on exit.
    graph
        Output a Graphviz DOT file of the node graph.

    Examples
    --------
    $ tinman playback myapp:consumer  # uses latest ~/.tinman/logs/
    $ tinman playback myapp:consumer --from logs/
    $ tinman playback myapp:consumer --from logs/ --speed 1.0
    $ tinman playback myapp:transform --from logs/ --log-dir processed/
    $ tinman playback myapp:consumer --stats
    $ tinman playback myapp:consumer --graph graph.dot
    """
    # Determine playback directory
    if from_ is None:
        from_ = _get_latest_log_dir()
        if from_ is None:
            print("Error: No log directory specified and no logs found in ~/.tinman/logs/", file=sys.stderr)
            sys.exit(1)
    
    if pickle:
        enable_pickle_codec()
    node_list = load_nodes(nodes)
    
    # Build graph from user nodes
    g = Graph.of(*node_list)
    
    # Check if logs channel will exist (from capture or recorded)
    has_recorded_logs = (from_ / f"{log_channel}.blog").exists()
    has_logs_channel = capture_logs or has_recorded_logs
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    # Set root logger level to allow log capture at the specified level
    if capture_logs:
        logging.getLogger().setLevel(level)
    with log_capture_context(
        capture_logs,
        channel=log_channel,
        level=level,
        use_virtual_time=False,  # Timer factory is installed by with_playback
    ) as log_nodes:
        # Add log capture nodes
        g.nodes.extend(log_nodes)
        
        # Add log printer if logs channel exists
        if has_logs_channel:
            g.nodes.append(create_log_printer(log_channel))
        
        # Add stats if requested
        if stats:
            g.nodes.append(create_stats_node(daemon=True))
        
        # Add recording if log_dir specified
        if log_dir is not None:
            g.nodes.append(create_recording_node(log_dir, g.nodes))
        
        # Apply playback transform
        g = with_playback(from_, speed=speed, use_virtual_time_logs=use_virtual_time)(g)
        
        # If --graph specified, output DOT
        if graph is not None:
            graph.write_text(g.to_dot())
        
        # Run
        asyncio.run(g.run())


@app.command
def logs(
    *,
    from_: Annotated[Path | None, cyclopts.Parameter(name=["--from", "-f"])] = None,
    channel: str = "logs",
    speed: float = float('inf'),
    node: str | None = None,
    level: str = "DEBUG",
) -> None:
    """Play back recorded log messages.

    This command reads log entries from a recorded log directory and prints them
    with colored formatting. Useful for reviewing what happened during a previous run.

    Parameters
    ----------
    from_
        Directory containing log files. Default: latest ~/.tinman/logs/
    channel
        Name of the logs channel. Default: "logs".
    speed
        Playback speed. Use 'inf' (default) for instant playback, or 1.0 for
        real-time playback that respects original timing.
    node
        Filter to show only logs from a specific node name.
    level
        Minimum log level to show (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Default: DEBUG (show all).

    Examples
    --------
    $ tinman logs  # uses latest ~/.tinman/logs/
    $ tinman logs --from logs/
    $ tinman logs --from logs/ --speed 1.0
    $ tinman logs --from logs/ --node data_processor
    $ tinman logs --from logs/ --level WARNING
    """
    # Determine playback directory
    if from_ is None:
        from_ = _get_latest_log_dir()
        if from_ is None:
            print("Error: No log directory specified and no logs found in ~/.tinman/logs/", file=sys.stderr)
            sys.exit(1)
    
    from .logging import LogEntry, create_log_printer
    from .oblog import ObLogReader
    
    min_level = getattr(logging, level.upper(), logging.DEBUG)
    
    async def _logs() -> None:
        log_file = from_ / f"{channel}.blog"
        if not log_file.exists():
            print(f"No log file found: {log_file}", file=sys.stderr)
            sys.exit(1)
        
        # For instant playback, just read and print directly
        if speed == float('inf'):
            reader = ObLogReader(from_)
            from datetime import datetime
            from rich.console import Console
            from rich.text import Text
            import logging as log_module
            
            LEVEL_STYLES = {
                log_module.DEBUG: "dim",
                log_module.INFO: "green",
                log_module.WARNING: "yellow",
                log_module.ERROR: "red",
                log_module.CRITICAL: "red bold",
            }
            console = Console()
            
            async for _ts, entry in reader.read_channel(channel):
                entry: LogEntry  # type: ignore[no-redef]
                # Apply filters
                if node is not None and entry.node_name != node:
                    continue
                if entry.level < min_level:
                    continue
                    
                level_name = log_module.getLevelName(entry.level)
                style = LEVEL_STYLES.get(entry.level, "white")
                timestamp = datetime.fromtimestamp(entry.timestamp_ns / 1_000_000_000)
                time_str = timestamp.strftime("%H:%M:%S.%f")
                
                text = Text()
                text.append(time_str, style="dim")
                text.append(" ")
                text.append(f"[{level_name:8}]", style=style)
                if entry.node_name:
                    text.append(f"[{entry.node_name}] ", style="cyan")
                else:
                    text.append(" ")
                text.append(f"{entry.name}: {entry.message}")
                console.print(text)
                
                if entry.exc_text:
                    console.print(Text(entry.exc_text, style="red"))
        else:
            # For timed playback, use the full playback infrastructure
            from .pubsub import In
            from .runtime import NodeSpec
            
            # Create a filtered log printer
            from datetime import datetime
            from rich.console import Console
            from rich.text import Text
            import logging as log_module
            
            LEVEL_STYLES = {
                log_module.DEBUG: "dim",
                log_module.INFO: "green",
                log_module.WARNING: "yellow",
                log_module.ERROR: "red",
                log_module.CRITICAL: "red bold",
            }
            console = Console()
            
            async def filtered_log_printer(logs_in: In[LogEntry]) -> None:
                async for entry in logs_in:
                    # Apply filters
                    if node is not None and entry.node_name != node:
                        continue
                    if entry.level < min_level:
                        continue
                    
                    level_name = log_module.getLevelName(entry.level)
                    style = LEVEL_STYLES.get(entry.level, "white")
                    timestamp = datetime.fromtimestamp(entry.timestamp_ns / 1_000_000_000)
                    time_str = timestamp.strftime("%H:%M:%S.%f")
                    
                    text = Text()
                    text.append(time_str, style="dim")
                    text.append(" ")
                    text.append(f"[{level_name:8}]", style=style)
                    if entry.node_name:
                        text.append(f"[{entry.node_name}] ", style="cyan")
                    else:
                        text.append(" ")
                    text.append(f"{entry.name}: {entry.message}")
                    console.print(text)
                    
                    if entry.exc_text:
                        console.print(Text(entry.exc_text, style="red"))
            
            printer_spec = NodeSpec(
                node_fn=filtered_log_printer,
                inputs={"logs_in": (channel, 100)},
                outputs={},
                daemon=True,
            )
            
            # Use Graph API with playback transform
            g = Graph.of(printer_spec)
            g = with_playback(from_, speed=speed)(g)
            asyncio.run(g.run())
            return
    
    asyncio.run(_logs())


@app.command
def stats(
    *,
    from_: Annotated[Path | None, cyclopts.Parameter(name=["--from", "-f"])] = None,
    pickle: bool = False,
) -> None:
    """Print channel statistics from recorded logs.

    This is a standalone command that plays back all channels in a log directory
    and prints statistics (message counts and Hz) without running any consumer nodes.

    Parameters
    ----------
    from_
        Directory containing log files to analyze. Default: latest ~/.tinman/logs/
    pickle
        Enable PickleCodec for arbitrary Python objects (security risk with untrusted data).

    Examples
    --------
    $ tinman stats  # uses latest ~/.tinman/logs/
    $ tinman stats --from logs/
    $ tinman stats -f webcam_logs/
    """
    # Determine playback directory
    if from_ is None:
        from_ = _get_latest_log_dir()
        if from_ is None:
            print("Error: No log directory specified and no logs found in ~/.tinman/logs/", file=sys.stderr)
            sys.exit(1)
    
    from .stats import run_stats
    
    if pickle:
        enable_pickle_codec()
    
    asyncio.run(run_stats(from_))


def main() -> None:
    """Entry point for the CLI."""
    # Add current directory to path so local modules can be imported
    if "" not in sys.path:
        sys.path.insert(0, "")
    app()


if __name__ == "__main__":
    main()
