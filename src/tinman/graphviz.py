"""Generate Graphviz DOT files from node graphs.

This module provides functions to visualize tinman node graphs as DOT files
that can be rendered with Graphviz.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .runtime import NodeSpec


def generate_dot(
    specs: list[NodeSpec],
    user_node_names: set[str] | None = None,
) -> str:
    """Generate a Graphviz DOT representation of a node graph.
    
    Args:
        specs: List of NodeSpec objects describing the graph.
        user_node_names: Names of user-provided nodes (vs system-generated).
                        If None, all nodes are treated as user nodes.
    
    Returns:
        DOT format string that can be rendered with Graphviz.
    """
    if user_node_names is None:
        user_node_names = {spec.name for spec in specs if spec.name is not None}
    
    lines = [
        "digraph tinman_graph {",
        "    // Graph settings",
        "    rankdir=LR;",
        "    splines=spline;",
        "    nodesep=0.6;",
        "    ranksep=1.0;",
        '    fontname="Helvetica";',
        "",
        '    node [fontname="Helvetica", fontsize=11];',
        '    edge [fontname="Helvetica", fontsize=9];',
        "",
    ]
    
    # Collect all channels and their producers
    channel_producers: dict[str, str] = {}  # channel_name -> node_name
    channel_consumers: dict[str, list[str]] = {}  # channel_name -> [node_names]
    all_channels_nodes: list[str] = []  # nodes that consume all channels
    
    for spec in specs:
        node_name = spec.name or spec.node_fn.__name__
        
        # Track outputs (this node produces these channels)
        for _param, (channel_name, _codec) in spec.outputs.items():
            channel_producers[channel_name] = node_name
        
        # Track inputs (this node consumes these channels)
        for _param, (channel_name, _queue_size) in spec.inputs.items():
            if channel_name not in channel_consumers:
                channel_consumers[channel_name] = []
            channel_consumers[channel_name].append(node_name)
        
        # Track all_channels consumers
        if spec.all_channels_param is not None:
            all_channels_nodes.append(node_name)
    
    # Determine which channels are "user" channels
    # A channel is a "user channel" if:
    # - It's produced by a user node, OR
    # - It's consumed by a user node (even if produced by playback)
    # But NOT if it's only consumed by system nodes (like "logs")
    all_channels = set(channel_producers.keys()) | set(channel_consumers.keys())
    user_channels: set[str] = set()
    for ch in all_channels:
        # Produced by user node?
        if ch in channel_producers and channel_producers[ch] in user_node_names:
            user_channels.add(ch)
            continue
        # Consumed by any user node?
        consumers = channel_consumers.get(ch, [])
        if any(consumer in user_node_names for consumer in consumers):
            user_channels.add(ch)
    system_channels = all_channels - user_channels
    
    # === USER NODES CLUSTER ===
    lines.append("    // === USER NODES CLUSTER ===")
    lines.append("    subgraph cluster_user {")
    lines.append('        style=filled;')
    lines.append('        fillcolor="#f8fbfe";')
    lines.append('        color="#2874a6";')
    lines.append("        penwidth=1.5;")
    lines.append("")
    
    # User nodes
    lines.append('        node [shape=box, style="filled,rounded", fillcolor="#d4e6f1", color="#2874a6", penwidth=2];')
    for spec in specs:
        node_name = spec.name or spec.node_fn.__name__
        if node_name in user_node_names:
            label = node_name
            if spec.daemon:
                label += "\\n(daemon)"
            lines.append(f'        {_dot_id(node_name, "n")} [label="{label}"];')
    lines.append("")
    
    # User channels (inside cluster)
    if user_channels:
        lines.append('        node [shape=ellipse, style=filled, fillcolor="#d5f5e3", color="#1e8449", penwidth=1.5];')
        for channel in sorted(user_channels):
            lines.append(f'        {_dot_id(channel, "c")} [label="{channel}"];')
        lines.append("")
    
    lines.append("    }")
    lines.append("")
    
    # === SYSTEM CLUSTER (channels + nodes) ===
    system_node_specs = [s for s in specs if (s.name or s.node_fn.__name__) not in user_node_names]
    if system_channels or system_node_specs:
        lines.append("    // === SYSTEM CLUSTER ===")
        lines.append("    subgraph cluster_system {")
        lines.append('        style=filled;')
        lines.append('        fillcolor="#f9f9f9";')
        lines.append('        color="#7f8c8d";')
        lines.append("        penwidth=1.5;")
        lines.append("")
        
        # System channels
        if system_channels:
            lines.append('        node [shape=ellipse, style=filled, fillcolor="#e8e8e8", color="#7f8c8d", penwidth=1.5];')
            for channel in sorted(system_channels):
                lines.append(f'        {_dot_id(channel, "c")} [label="{channel}"];')
            lines.append("")
        
        # System nodes
        if system_node_specs:
            lines.append('        node [shape=box, style="filled,rounded", fillcolor="#f4f4f4", color="#7f8c8d", penwidth=1.5];')
            for spec in system_node_specs:
                node_name = spec.name or spec.node_fn.__name__
                label = node_name
                if spec.daemon:
                    label += "\\n(daemon)"
                lines.append(f'        {_dot_id(node_name, "n")} [label="{label}"];')
            lines.append("")
        
        lines.append("    }")
        lines.append("")
    
    # === EDGES ===
    lines.append("    // === EDGES ===")
    
    # User node edges (solid)
    lines.append('    edge [color="#34495e", penwidth=1.5];')
    for spec in specs:
        node_name = spec.name or spec.node_fn.__name__
        if node_name not in user_node_names:
            continue
        
        # Outputs: node -> channel
        for _param, (channel_name, _codec) in spec.outputs.items():
            lines.append(f"    {_dot_id(node_name, 'n')} -> {_dot_id(channel_name, 'c')};")
        
        # Inputs: channel -> node
        for _param, (channel_name, _queue_size) in spec.inputs.items():
            lines.append(f"    {_dot_id(channel_name, 'c')} -> {_dot_id(node_name, 'n')};")
    lines.append("")
    
    # System node edges (dashed)
    if system_node_specs or all_channels_nodes:
        lines.append("    // System node connections")
        lines.append('    edge [color="#95a5a6", style=dashed, penwidth=1];')
        
        for spec in specs:
            node_name = spec.name or spec.node_fn.__name__
            
            # System node regular inputs/outputs
            if node_name not in user_node_names:
                for _param, (channel_name, _codec) in spec.outputs.items():
                    lines.append(f"    {_dot_id(node_name, 'n')} -> {_dot_id(channel_name, 'c')};")
                for _param, (channel_name, _queue_size) in spec.inputs.items():
                    lines.append(f"    {_dot_id(channel_name, 'c')} -> {_dot_id(node_name, 'n')};")
            
            # All-channels connections (any node can have this)
            if spec.all_channels_param is not None:
                for channel in sorted(all_channels):
                    lines.append(f"    {_dot_id(channel, 'c')} -> {_dot_id(node_name, 'n')};")
    
    lines.append("}")
    return "\n".join(lines)


def _dot_id(name: str, prefix: str = "") -> str:
    """Convert a name to a valid DOT identifier.
    
    Args:
        name: The name to convert.
        prefix: Optional prefix to distinguish nodes from channels.
    """
    # Replace problematic characters
    safe = name.replace("-", "_").replace(".", "_").replace(" ", "_")
    safe = safe.replace("[", "_").replace("]", "_").replace("(", "_").replace(")", "_")
    # If it starts with a digit, prefix with underscore
    if safe and safe[0].isdigit():
        safe = "_" + safe
    if prefix:
        safe = f"{prefix}_{safe}"
    return safe


def write_dot(
    specs: list[NodeSpec],
    output: Path | TextIO,
    user_node_names: set[str] | None = None,
) -> None:
    """Write a DOT file for a node graph.
    
    Args:
        specs: List of NodeSpec objects describing the graph.
        output: Path to write to, or file-like object.
        user_node_names: Names of user-provided nodes (vs system-generated).
    """
    dot = generate_dot(specs, user_node_names)
    
    if isinstance(output, Path):
        output.write_text(dot)
    else:
        output.write(dot)
