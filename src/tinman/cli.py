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
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import cyclopts

from .launcher import playback as _playback
from .launcher import run as _run
from .runtime import NodeSpec, get_node_specs
from .oblog import enable_pickle_codec
from .logging import log_capture_context, create_log_printer
from .stats import create_stats_node

app = cyclopts.App(
    name="tinman",
    help="A minimal async framework for data pipelines with logging and playback.",
)


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
    pickle: bool = False,
    capture_logs: bool = True,
    log_level: str = "INFO",
    log_channel: str = "logs",
    stats: bool = False,
) -> None:
    """Run a node graph.

    Parameters
    ----------
    nodes
        One or more node specs in 'module:node1,node2' format.
        Multiple specs are combined into a single graph.
    log_dir
        Directory to log all output channels.
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

    Examples
    --------
    $ tinman run myapp.nodes:producer,consumer
    $ tinman run myapp.sensors:camera myapp.processors:detector --log-dir logs/
    $ tinman run myapp:nodes --log-dir logs/ --capture-logs
    $ tinman run myapp:nodes --log-dir logs/ --capture-logs --log-level DEBUG
    $ tinman run myapp:nodes --stats
    """
    if pickle:
        enable_pickle_codec()
    node_list = load_nodes(nodes)
    
    # Add stats node if requested
    stats_nodes = [create_stats_node(daemon=True)] if stats else []
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    with log_capture_context(capture_logs, channel=log_channel, level=level) as log_nodes:
        # Add log printer if we're capturing logs
        printer_nodes = [create_log_printer(log_channel)] if capture_logs else []
        asyncio.run(_run([*node_list, *log_nodes, *printer_nodes, *stats_nodes], log_dir=log_dir))  # type: ignore[arg-type]


@app.command
def playback(
    nodes: tuple[str, ...],
    *,
    from_: Annotated[Path, cyclopts.Parameter(name=["--from", "-f"])],
    speed: float = float("inf"),
    log_dir: Path | None = None,
    pickle: bool = False,
    capture_logs: bool = True,
    log_level: str = "INFO",
    log_channel: str = "logs",
    use_virtual_time: bool = True,
    stats: bool = False,
) -> None:
    """Play back recorded logs through nodes.

    Parameters
    ----------
    nodes
        One or more node specs in 'module:node1,node2' format.
        These consume channels from the recorded logs.
    from_
        Directory containing log files to play back.
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

    Examples
    --------
    $ tinman playback myapp:consumer --from logs/
    $ tinman playback myapp:consumer --from logs/ --speed 1.0
    $ tinman playback myapp:transform --from logs/ --log-dir processed/
    $ tinman playback myapp:consumer --from logs/ --capture-logs
    $ tinman playback myapp:consumer --from logs/ --stats
    """
    if pickle:
        enable_pickle_codec()
    node_list = load_nodes(nodes)
    
    # Add stats node if requested
    stats_nodes = [create_stats_node(daemon=True)] if stats else []
    
    # Check if logs channel will exist (from capture or recorded)
    has_recorded_logs = (from_ / f"{log_channel}.blog").exists()
    has_logs_channel = capture_logs or has_recorded_logs
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    with log_capture_context(
        capture_logs,
        channel=log_channel,
        level=level,
        use_virtual_time=use_virtual_time,
    ) as log_nodes:
        # Add log printer if logs channel exists
        printer_nodes = [create_log_printer(log_channel)] if has_logs_channel else []
        asyncio.run(_playback(
            [*node_list, *log_nodes, *printer_nodes, *stats_nodes],
            playback_dir=from_,
            speed=speed,
            log_dir=log_dir,
        ))  # type: ignore[arg-type]


@app.command
def stats(
    *,
    from_: Annotated[Path, cyclopts.Parameter(name=["--from", "-f"])],
    pickle: bool = False,
) -> None:
    """Print channel statistics from recorded logs.

    This is a standalone command that plays back all channels in a log directory
    and prints statistics (message counts and Hz) without running any consumer nodes.

    Parameters
    ----------
    from_
        Directory containing log files to analyze.
    pickle
        Enable PickleCodec for arbitrary Python objects (security risk with untrusted data).

    Examples
    --------
    $ tinman stats --from logs/
    $ tinman stats -f webcam_logs/
    """
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
