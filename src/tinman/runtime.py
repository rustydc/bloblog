"""Low-level node graph execution engine.

This module contains the core runtime for executing graphs of async nodes.
It handles channel wiring, subscription management, and concurrent execution.

For high-level utilities like logging and playback, see run.py.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from .oblog import Codec, PickleCodec
from .pubsub import In, Out
from .timer import Timer, ScaledTimer


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
    """

    node_fn: Callable
    inputs: dict[str, tuple[str, int]]  # param_name -> (channel_name, queue_size)
    outputs: dict[str, tuple[str, Codec]]  # param_name -> (channel_name, codec)
    all_channels_param: str | None = None  # Parameter name for dict[str, In] injection
    timer_param: str | None = None  # Parameter name for Timer injection


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
        List of NodeSpec objects, one per node.

    Raises:
        ValueError: If signature is invalid or codecs are missing.

    Example:
        >>> specs = get_node_specs([producer, consumer])
        >>> for spec in specs:
        ...     for _, (ch, codec) in spec.outputs.items():
        ...         print(f"{spec.node_fn.__name__} produces {ch}")
    """
    specs = []
    for node in nodes:
        if isinstance(node, NodeSpec):
            # Already a spec, pass through
            specs.append(node)
        else:
            # Parse callable
            inputs, outputs, all_channels_param, timer_param = _parse_node_signature(node)
            specs.append(NodeSpec(node, inputs, outputs, all_channels_param, timer_param))
    return specs


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
) -> None:
    """Run a graph of nodes, wiring their channels and executing them concurrently.

    This is the core execution engine that wires together async nodes based on
    their channel annotations, then executes them concurrently.

    Args:
        nodes: List of async callables (functions, bound methods, or NodeSpec objects).
               Each should have annotated parameters for channels:
               - Inputs: Annotated[In[T], "channel_name"]
               - Outputs: Annotated[Out[T], "channel_name", codec_instance]
               - All channels: dict[str, In] (for monitoring/logging nodes)
               - Timer: Timer (for time access)
        timer: Optional Timer instance to inject into nodes that request it.
               If None, a default ScaledTimer (real-time) is created.

    Raises:
        TypeError: If nodes are not async callables.
        ValueError: If channel configuration is invalid (e.g., missing input producers).

    Example:
        >>> # Simple pipeline
        >>> await run_nodes([producer, consumer])

        >>> # With custom timer
        >>> clock = VirtualClock()
        >>> timer = FastForwardTimer(clock)
        >>> await run_nodes([producer, consumer], timer=timer)
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
    # (inputs, outputs, all_channels_param, timer_param, kwargs)
    node_info: dict[Callable, tuple[dict, dict, str | None, str | None, dict[str, Any]]] = {}
    for spec in specs:
        node_info[spec.node_fn] = (spec.inputs, spec.outputs, spec.all_channels_param, spec.timer_param, {})

    # Build runtime channels for all node outputs
    runtime_channels: dict[str, _ChannelRuntime] = {}

    for spec in specs:
        # Create runtime channels for node outputs
        for _param_name, (channel_name, codec) in spec.outputs.items():
            if channel_name not in runtime_channels:
                runtime_channels[channel_name] = _ChannelRuntime(channel_name, codec)

    # Pre-build kwargs for all nodes to avoid race conditions
    # All subscriptions must be created BEFORE any node starts running
    for _node_fn, (inputs, outputs, all_channels_param, timer_param, kwargs) in node_info.items():
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
        _, outputs, _, _, kwargs = node_info[node_fn]
        try:
            await node_fn(**kwargs)
        finally:
            # Close output channels
            for _param_name, (channel_name, _) in outputs.items():
                await runtime_channels[channel_name].out.close()

    # Execute all nodes concurrently
    async with asyncio.TaskGroup() as tg:
        for spec in specs:
            tg.create_task(run_node_and_close(spec.node_fn))
