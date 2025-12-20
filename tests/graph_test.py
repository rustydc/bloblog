"""Tests for the graph module."""

from typing import Annotated

import pytest

from tinman import In, Out
from tinman.runtime import NodeSpec, get_node_specs
from tinman.graphviz import generate_dot


async def producer(output: Annotated[Out[str], "messages"]):
    """Producer node."""
    pass


async def consumer(input: Annotated[In[str], "messages"]):
    """Consumer node."""
    pass


async def logging_node(channels: dict[str, In]):
    """Logging node that receives all channels."""
    pass


class TestGenerateDot:
    def test_simple_graph(self):
        """Test generating DOT for a simple producer-consumer graph."""
        specs = get_node_specs([producer, consumer])
        dot = generate_dot(specs)
        
        # Should contain the node definitions
        assert 'n_producer [label="producer"]' in dot
        assert 'n_consumer [label="consumer"]' in dot
        
        # Should contain the channel
        assert 'c_messages [label="messages"]' in dot
        
        # Should contain edges
        assert "n_producer -> c_messages" in dot
        assert "c_messages -> n_consumer" in dot
        
        # Should be a valid digraph
        assert dot.startswith("digraph tinman_graph {")
        assert dot.endswith("}")
    
    def test_user_vs_system_nodes(self):
        """Test that user nodes are in cluster, system nodes outside."""
        specs = get_node_specs([producer, consumer, logging_node])
        # Only producer and consumer are user nodes
        user_names = {"producer", "consumer"}
        dot = generate_dot(specs, user_node_names=user_names)
        
        # Check cluster contains user nodes
        assert "subgraph cluster_user" in dot
        
        # Logging node should be in system cluster
        assert "subgraph cluster_system" in dot
        assert 'n_logging_node [label="logging_node"]' in dot
    
    def test_all_channels_node(self):
        """Test that all_channels nodes get edges from all channels."""
        specs = get_node_specs([producer, consumer, logging_node])
        user_names = {"producer", "consumer"}
        dot = generate_dot(specs, user_node_names=user_names)
        
        # Logging node should have dashed edge from messages channel
        assert "c_messages -> n_logging_node" in dot
    
    def test_node_channel_name_collision(self):
        """Test that nodes and channels with same name don't collide."""
        # The "uppercase" example has both a node and channel named "uppercase"
        async def uppercase(
            input: Annotated[In[str], "messages"],
            output: Annotated[Out[str], "uppercase"],
        ):
            pass
        
        async def consumer_node(input: Annotated[In[str], "uppercase"]):
            pass
        
        specs = get_node_specs([uppercase, consumer_node])
        dot = generate_dot(specs)
        
        # Node and channel should have different IDs
        assert "n_uppercase" in dot  # node
        assert "c_uppercase" in dot  # channel
    
    def test_special_characters_in_names(self):
        """Test that special characters are handled."""
        spec = NodeSpec(
            node_fn=producer,
            inputs={},
            outputs={"out": ("data", None)},  # type: ignore
            name="node[1]",
        )
        dot = generate_dot([spec], user_node_names={"node[1]"})
        
        # Brackets should be replaced with underscores
        assert "n_node_1_" in dot
