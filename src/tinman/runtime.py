"""Low-level node graph execution engine.

This module contains the core runtime for executing graphs of async nodes.
It handles channel wiring, subscription management, and concurrent execution.

Shutdown behavior:
- External shutdown (SIGINT/SIGTERM): All nodes are cancelled gracefully
- Natural completion: When all non-daemon nodes complete, daemons are cancelled
- Error in any node: All nodes are cancelled (TaskGroup semantics)

Nodes should be cancellation-safe. Use try/finally for cleanup:

    async def my_node(input: In[T], output: Out[T]) -> None:
        try:
            async for item in input:
                await output.publish(process(item))
        except asyncio.CancelledError:
            # Optional: log cancellation
            raise  # Always re-raise

For high-level utilities like logging and playback, see launcher.py.
"""

from __future__ import annotations

import asyncio
import inspect
import signal
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from .oblog import Codec, PickleCodec
from .pubsub import In, Out
from .timer import Timer, ScaledTimer


# ContextVar for tracking the currently executing node's name
_current_node_name: ContextVar[str | None] = ContextVar("current_node_name", default=None)


def get_current_node_name() -> str | None:
    """Get the name of the currently executing node.
    
    Returns the node name when called from within a running node,
    or None if called outside of node execution.
    
    This can be useful for logging, debugging, or conditional behavior
    based on which node is running.
    
    Example:
        >>> async def my_node(input: Annotated[In[str], "data"]) -> None:
        ...     print(f"Running in node: {get_current_node_name()}")
        ...     async for item in input:
        ...         process(item)
    """
    return _current_node_name.get()


# Default timeout for graceful shutdown before force-cancelling
DEFAULT_SHUTDOWN_TIMEOUT = 5.0


@dataclass
class NodeSpec:
    """Specification for a node's inputs and outputs.

    This is the fundamental metadata structure describing how a node
    connects to the graph through channels.

    Attributes:
        node_fn: The async callable that implements the node
        inputs: Maps parameter name -> (channel_name, queue_size)
        outputs: Maps parameter name -> (channel_name, codec)
        all_channels_param: Parameter name for dict[str, In] injection (or None)
        timer_param: Parameter name for Timer injection (or None)
        daemon: If True, node is cancelled when all non-daemon nodes complete.
                Useful for auxiliary nodes like log capture that should not
                block shutdown. Daemon nodes are also cancelled on external
                shutdown (SIGINT/SIGTERM) along with all other nodes.
        name: Human-readable name for the node. If None, auto-derived from
              the function name. Used in error messages, logging context,
              and for selective playback.
    """

    node_fn: Callable
    inputs: dict[str, tuple[str, int]]  # param_name -> (channel_name, queue_size)
    outputs: dict[str, tuple[str, Codec]]  # param_name -> (channel_name, codec)
    all_channels_param: str | None = None  # Parameter name for dict[str, In] injection
    timer_param: str | None = None  # Parameter name for Timer injection
    daemon: bool = False  # If True, cancelled when non-daemon nodes complete
    name: str | None = None  # Human-readable name (auto-derived if None)


class ShutdownRequested(Exception):
    """Exception for signal-based shutdown (kept for potential future use).
    
    Note: run_nodes() now swallows this and returns normally on SIGINT/SIGTERM,
    treating signal-based shutdown as a clean exit rather than an error.
    
    This class is still exported in case users want to manually trigger
    shutdown semantics or for future extensions.
    """
    pass


def daemon[F: Callable](node_fn: F) -> F:
    """Mark a node function as a daemon.
    
    Daemon nodes are cancelled when all non-daemon nodes complete.
    Use this for auxiliary nodes like log consumers that should not
    block shutdown.
    
    Example:
        >>> @daemon
        ... async def log_printer(logs: Annotated[In[LogEntry], "logs"]):
        ...     async for entry in logs:
        ...         print(entry.message)
    """
    node_fn._tinman_daemon = True  # type: ignore[attr-defined]
    return node_fn


class _ChannelRuntime[T]:
    """Internal runtime channel linking publishers and subscribers.
    
    Each channel has one publisher (Out) and zero or more subscribers (In).
    The runtime creates these and manages their lifecycle.
    """

    def __init__(self, name: str, codec: Codec[T]) -> None:
        self.name = name
        self.codec = codec
        self.out = Out[T]()


def _parse_node_signature(
    node_fn: Callable,
) -> tuple[dict[str, tuple[str, int]], dict[str, tuple[str, Codec]], str | None, str | None]:
    """Parse a node function's signature to extract input/output channels.

    This is the core introspection function that understands the channel
    annotation DSL:
    - Annotated[In[T], "channel_name"] for inputs
    - Annotated[Out[T], "channel_name", codec] for outputs
    - dict[str, In] for all-channels injection (e.g., logging nodes)
    - Timer for timer injection

    Args:
        node_fn: The node function or bound method to inspect.

    Returns:
        (inputs, outputs, all_channels_param, timer_param) where:
        - inputs maps param_name -> (channel_name, queue_size)
        - outputs maps param_name -> (channel_name, codec)
        - all_channels_param is the param name for dict[str, In] injection (or None)
        - timer_param is the param name for Timer injection (or None)

    Raises:
        ValueError: If signature is invalid or codecs are missing from outputs.
    """
    sig = inspect.signature(node_fn)
    hints = get_type_hints(node_fn, include_extras=True)

    inputs: dict[str, tuple[str, int]] = {}
    outputs: dict[str, tuple[str, Codec]] = {}
    all_channels_param: str | None = None
    timer_param: str | None = None

    for param_name, _param in sig.parameters.items():
        if param_name == "self":
            continue

        if param_name not in hints:
            raise ValueError(f"Parameter '{param_name}' in {node_fn.__name__}() has no type hint")

        hint = hints[param_name]

        # Check for Timer parameter
        if hint is Timer or (get_origin(hint) is type and issubclass(get_args(hint)[0] if get_args(hint) else hint, Timer)):
            if timer_param is not None:
                raise ValueError(
                    f"Node {node_fn.__name__}() has multiple Timer parameters: "
                    f"'{timer_param}' and '{param_name}'"
                )
            timer_param = param_name
            continue

        # Check for dict[str, In] (all channels injection)
        if get_origin(hint) is dict:
            args = get_args(hint)
            # Check if it's dict[str, In] or dict[str, In[T]]
            if len(args) == 2 and args[0] is str:
                # args[1] could be In itself or In[T]
                if args[1] is In or get_origin(args[1]) is In:
                    if all_channels_param is not None:
                        raise ValueError(
                            f"Node {node_fn.__name__}() has multiple dict[str, In] parameters: "
                            f"'{all_channels_param}' and '{param_name}'"
                        )
                    all_channels_param = param_name
                    continue

        # Must be Annotated for regular channels
        if get_origin(hint) is not Annotated:
            raise ValueError(
                f"Parameter '{param_name}' in {node_fn.__name__}() "
                f'must be Annotated[In[T], "channel"], '
                f'Annotated[Out[T], "channel", codec], dict[str, In], or Timer'
            )

        # Parse Annotated[In[T] | Out[T], channel_name, codec?]
        args = get_args(hint)
        if len(args) < 2:
            raise ValueError(
                f"Parameter '{param_name}' must be Annotated[In[T], \"channel\"] "
                f'or Annotated[Out[T], "channel", codec]'
            )

        base_type = args[0]  # In[T] or Out[T]
        channel_name = args[1]  # "camera"
        origin = get_origin(base_type)

        if origin is In:
            # Inputs can have 2 args (default queue size) or 3 args (with queue size)
            queue_size = 10  # default
            if len(args) == 2:
                # Annotated[In[T], "channel"]
                pass
            elif len(args) == 3:
                # Annotated[In[T], "channel", queue_size]
                queue_size = args[2]
                if not isinstance(queue_size, int) or queue_size <= 0:
                    raise ValueError(
                        f"Input parameter '{param_name}' queue size must be a positive integer, "
                        f"got {queue_size}"
                    )
            else:
                raise ValueError(
                    f"Input parameter '{param_name}' must be Annotated[In[T], \"channel\"] "
                    f"or Annotated[In[T], \"channel\", queue_size]"
                )
            inputs[param_name] = (channel_name, queue_size)

        elif origin is Out:
            # Outputs can have 2 args (use PickleCodec) or 3 args (explicit codec)
            if len(args) == 2:
                # No codec specified, use PickleCodec as default
                codec = PickleCodec()
            elif len(args) == 3:
                codec = args[2]
                if not isinstance(codec, Codec):
                    raise ValueError(
                        f"Output parameter '{param_name}' codec must be a Codec instance, "
                        f"got {type(codec)}"
                    )
            else:
                raise ValueError(
                    f"Output parameter '{param_name}' must be "
                    f'Annotated[Out[T], "channel"] or Annotated[Out[T], "channel", codec]'
                )
            outputs[param_name] = (channel_name, codec)

        else:
            raise ValueError(f"Parameter '{param_name}' must be In[T] or Out[T], got {base_type}")

    return inputs, outputs, all_channels_param, timer_param


def get_node_specs(nodes: Sequence[Callable | NodeSpec]) -> list[NodeSpec]:
    """Parse node signatures to extract channel wiring specifications.

    This function introspects node function signatures to determine which
    channels each node reads from/writes to. Use this to:
    - Set up playback nodes for missing inputs
    - Set up logging nodes with channel metadata
    - Validate node connectivity before running

    Args:
        nodes: List of node callables (functions or bound methods) or NodeSpec objects.
               NodeSpec objects are passed through as-is (useful for synthetic nodes).

    Returns:
        List of NodeSpec objects, one per node, with unique names assigned.

    Raises:
        ValueError: If signature is invalid or codecs are missing.

    Example:
        >>> specs = get_node_specs([producer, consumer])
        >>> for spec in specs:
        ...     print(f"{spec.name} produces {list(spec.outputs.values())}")
    """
    specs = []
    for node in nodes:
        if isinstance(node, NodeSpec):
            # Already a spec, pass through
            specs.append(node)
        else:
            # Parse callable
            inputs, outputs, all_channels_param, timer_param = _parse_node_signature(node)
            is_daemon = getattr(node, '_tinman_daemon', False)
            specs.append(NodeSpec(node, inputs, outputs, all_channels_param, timer_param, is_daemon))
    
    # Assign unique names to all specs
    _assign_unique_names(specs)
    return specs


def _assign_unique_names(specs: list[NodeSpec]) -> None:
    """Assign unique names to specs that don't have one.
    
    Names are derived from the function name. If multiple specs have the same
    derived name, they are disambiguated with _1, _2, etc. suffixes.
    
    Args:
        specs: List of NodeSpec objects to assign names to (modified in place).
    """
    # First pass: collect base names and count occurrences
    base_name_counts: dict[str, int] = {}
    base_names: list[str] = []
    
    for spec in specs:
        if spec.name is not None:
            # User-provided name - use as-is
            base_name = spec.name
        else:
            # Derive from function name
            base_name = spec.node_fn.__name__
        base_names.append(base_name)
        base_name_counts[base_name] = base_name_counts.get(base_name, 0) + 1
    
    # Second pass: assign unique names
    # Track how many of each base name we've seen so far
    base_name_seen: dict[str, int] = {}
    
    for i, spec in enumerate(specs):
        base_name = base_names[i]
        
        if base_name_counts[base_name] == 1:
            # Unique name, no suffix needed
            spec.name = base_name
        else:
            # Need to disambiguate
            seen = base_name_seen.get(base_name, 0)
            base_name_seen[base_name] = seen + 1
            spec.name = f"{base_name}_{seen + 1}"


def validate_nodes(nodes: Sequence[Callable | NodeSpec]) -> None:
    """Validate that node inputs/outputs form a valid graph.

    This checks:
    1. No duplicate output channels (each channel has exactly one producer)
    2. All input channels have a corresponding producer (no dangling inputs)
    3. Nodes with dict[str, In] injection are skipped from input validation

    Args:
        nodes: List of node callables or NodeSpec objects to validate.

    Raises:
        ValueError: If the graph is invalid with details about the problem.
    """
    # Convert all nodes to NodeSpec for uniform handling
    specs: list[NodeSpec] = []
    for node in nodes:
        if isinstance(node, NodeSpec):
            specs.append(node)
        else:
            # It's a callable - parse it
            inputs, outputs, all_channels_param, timer_param = _parse_node_signature(node)
            specs.append(NodeSpec(node, inputs, outputs, all_channels_param, timer_param))

    output_channels: dict[str, str] = {}  # channel_name -> node_name

    # Collect all outputs and check for duplicates
    for spec in specs:
        node_name = spec.node_fn.__name__
        for _param_name, (channel_name, _codec) in spec.outputs.items():
            if channel_name in output_channels:
                raise ValueError(
                    f"Output channel '{channel_name}' produced by multiple nodes: "
                    f"{output_channels[channel_name]} and {node_name}"
                )
            output_channels[channel_name] = node_name

    # Check all inputs have corresponding outputs
    for spec in specs:
        node_name = spec.node_fn.__name__
        # Skip validation if this node accepts all channels (like a logging node)
        if spec.all_channels_param:
            continue
        for _param_name, (channel_name, _queue_size) in spec.inputs.items():
            if channel_name not in output_channels:
                raise ValueError(
                    f"Input channel '{channel_name}' in {node_name}() has no producer node. "
                    f"Use create_playback_graph() to add playback nodes for recorded data."
                )


async def run_nodes(
    nodes: Sequence[Callable[..., Awaitable[Any]] | NodeSpec],
    timer: Timer | None = None,
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    install_signal_handlers: bool = True,
) -> None:
    """Run a graph of nodes, wiring their channels and executing them concurrently.

    This is the core execution engine that wires together async nodes based on
    their channel annotations, then executes them concurrently.

    Shutdown behavior:
    - SIGINT/SIGTERM: All nodes are cancelled gracefully, then returns normally
    - Natural completion: When all non-daemon nodes complete, daemons are cancelled
    - Any node raises: All nodes are cancelled, exception propagates
    - External cancellation: CancelledError propagates after cleanup

    Nodes should be cancellation-safe. Use try/finally for cleanup.

    Args:
        nodes: List of async callables (functions, bound methods, or NodeSpec objects).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name", codec_instance]
               - All channels: dict[str, In] (for monitoring/logging nodes)
               - Timer: Timer (for time access)
        timer: Optional Timer instance to inject into nodes that request it.
               If None, a default ScaledTimer (real-time) is created.
        shutdown_timeout: Seconds to wait for graceful shutdown before force-cancelling.
               Default is 5.0 seconds.
        install_signal_handlers: If True, install SIGINT/SIGTERM handlers for graceful
               shutdown. Set to False when running in contexts that manage their own
               signals (e.g., pytest, nested event loops). Default is True.

    Raises:
        TypeError: If nodes are not async callables.
        ValueError: If channel configuration is invalid (e.g., missing input producers).
        asyncio.CancelledError: If the caller cancels the run_nodes task.

    Example:
        >>> # Simple pipeline - Ctrl+C returns cleanly
        >>> await run_nodes([producer, consumer])

        >>> # With custom timer
        >>> clock = VirtualClock()
        >>> timer = FastForwardTimer(clock)
        >>> await run_nodes([producer, consumer], timer=timer)
        
        >>> # In tests (no signal handlers)
        >>> await run_nodes([producer, consumer], install_signal_handlers=False)
    """
    # Create default timer if not provided
    if timer is None:
        timer = ScaledTimer()

    # Validate all are async callables (or NodeSpec objects)
    for node in nodes:
        if isinstance(node, NodeSpec):
            # NodeSpec wraps a node_fn - validate that
            if not asyncio.iscoroutinefunction(node.node_fn):
                raise TypeError(
                    f"{node.node_fn} in NodeSpec must be an async function or method. "
                    f"If using a class, pass instance.run not the instance itself."
                )
        elif not asyncio.iscoroutinefunction(node):
            raise TypeError(
                f"{node} must be an async function or method. "
                f"If using a class, pass instance.run not the instance itself."
            )

    # Parse all node signatures to find inputs and outputs
    specs = get_node_specs(nodes)
    
    # Validate the graph structure
    validate_nodes(specs)
    
    # Build node info dictionary
    # (name, inputs, outputs, all_channels_param, timer_param, kwargs)
    # Note: spec.name is always set by get_node_specs via _assign_unique_names
    node_info: dict[Callable, tuple[str, dict, dict, str | None, str | None, dict[str, Any]]] = {}
    for spec in specs:
        assert spec.name is not None  # Guaranteed by _assign_unique_names
        node_info[spec.node_fn] = (spec.name, spec.inputs, spec.outputs, spec.all_channels_param, spec.timer_param, {})

    # Build runtime channels for all node outputs
    runtime_channels: dict[str, _ChannelRuntime] = {}

    for spec in specs:
        # Create runtime channels for node outputs
        for _param_name, (channel_name, codec) in spec.outputs.items():
            if channel_name not in runtime_channels:
                runtime_channels[channel_name] = _ChannelRuntime(channel_name, codec)

    # Pre-build kwargs for all nodes to avoid race conditions
    # All subscriptions must be created BEFORE any node starts running
    for _node_fn, (_name, inputs, outputs, all_channels_param, timer_param, kwargs) in node_info.items():
        for param_name, (channel_name, queue_size) in inputs.items():
            # Get the Out from runtime channels
            out = runtime_channels[channel_name].out
            kwargs[param_name] = out.sub(maxsize=queue_size)

        for param_name, (channel_name, _) in outputs.items():
            kwargs[param_name] = runtime_channels[channel_name].out

        # Handle dict[str, In] injection for logging/monitoring nodes
        if all_channels_param:
            all_channels_dict = {}
            for channel_name, runtime in runtime_channels.items():
                all_channels_dict[channel_name] = runtime.out.sub(maxsize=10)
            kwargs[all_channels_param] = all_channels_dict

        # Handle Timer injection
        if timer_param:
            kwargs[timer_param] = timer

    async def run_node_and_close(node_fn: Callable) -> None:
        """Run a node and close its output channels when done."""
        node_name, _, outputs, _, _, kwargs = node_info[node_fn]
        # Set the contextvar so the node can access its name
        token = _current_node_name.set(node_name)
        try:
            await node_fn(**kwargs)
        except asyncio.CancelledError:
            # Re-raise after closing outputs
            raise
        finally:
            _current_node_name.reset(token)
            # Close output channels
            for _param_name, (channel_name, _) in outputs.items():
                await runtime_channels[channel_name].out.close()

    # Separate daemon and non-daemon nodes
    daemon_specs = [s for s in specs if s.daemon]
    main_specs = [s for s in specs if not s.daemon]

    # Track all tasks for shutdown
    all_tasks: list[asyncio.Task] = []
    shutdown_event = asyncio.Event()
    
    async def graceful_shutdown(reason: str = "shutdown requested") -> None:
        """Signal shutdown - tasks will be cancelled by the main loop."""
        if shutdown_event.is_set():
            return  # Already shutting down
        shutdown_event.set()

    # Set up signal handlers if requested
    loop = asyncio.get_running_loop()
    
    def handle_signal() -> None:
        """Handle shutdown signals by triggering graceful shutdown."""
        if not shutdown_event.is_set():
            shutdown_event.set()

    if install_signal_handlers:
        try:
            # Use asyncio's signal handling - more reliable than signal.signal()
            # as it properly integrates with the event loop
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_signal)
        except (ValueError, OSError, NotImplementedError):
            # Can't set signal handlers (e.g., not main thread, or Windows)
            pass

    async def wait_with_shutdown(tasks: list[asyncio.Task]) -> None:
        """Wait for tasks to complete, but respond to shutdown signal."""
        if not tasks:
            return
            
        # Create a task that completes when shutdown is signaled
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())
        
        try:
            # Wait for either all tasks to complete OR shutdown signal
            task_set = set(tasks)
            task_set.add(shutdown_waiter)
            
            while task_set - {shutdown_waiter}:  # While there are real tasks
                done, pending = await asyncio.wait(
                    task_set,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Check if shutdown was signaled
                if shutdown_waiter in done:
                    # Cancel all remaining tasks
                    for task in pending:
                        if task is not shutdown_waiter:
                            task.cancel()
                    # Wait briefly for cancellation
                    if pending - {shutdown_waiter}:
                        await asyncio.wait(pending - {shutdown_waiter}, timeout=shutdown_timeout)
                    return
                
                # Remove completed tasks
                task_set = pending
                task_set.add(shutdown_waiter)  # Keep the shutdown waiter
                
                # Check for exceptions in done tasks
                for task in done:
                    if task is not shutdown_waiter and task.exception() is not None:
                        # Cancel remaining tasks and re-raise
                        for t in pending:
                            t.cancel()
                        if pending:
                            await asyncio.wait(pending, timeout=shutdown_timeout)
                        task.result()  # This will raise the exception
        finally:
            shutdown_waiter.cancel()
            try:
                await shutdown_waiter
            except asyncio.CancelledError:
                pass

    try:
        # Start daemon nodes
        for spec in daemon_specs:
            task = asyncio.create_task(run_node_and_close(spec.node_fn))
            all_tasks.append(task)

        # Start main nodes
        main_tasks: list[asyncio.Task] = []
        for spec in main_specs:
            task = asyncio.create_task(run_node_and_close(spec.node_fn))
            all_tasks.append(task)
            main_tasks.append(task)

        # Wait for main nodes to complete (or shutdown signal)
        await wait_with_shutdown(main_tasks)

        # Main nodes done - give daemons a brief moment to finish naturally
        # (they should see _CLOSED on their inputs and exit)
        daemon_tasks = [t for t in all_tasks if t not in main_tasks]
        if daemon_tasks:
            # Wait briefly for daemons to complete naturally (they should finish
            # quickly once their inputs close)
            done, pending = await asyncio.wait(daemon_tasks, timeout=0.1)
            
            # Cancel any still running
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                
        # Signal-based shutdown is a clean exit - just return normally
            
    except asyncio.CancelledError:
        # External cancellation (caller cancelled the task) - cancel all and re-raise
        shutdown_event.set()
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        raise
    except Exception:
        # On any error, clean up all tasks
        shutdown_event.set()
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        raise
    finally:
        # Remove signal handlers
        if install_signal_handlers:
            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)
            except (ValueError, OSError, NotImplementedError):
                pass