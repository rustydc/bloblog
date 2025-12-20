"""Tests for tinman.builder module."""

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock

import pytest

from tinman.launcher import Graph
from tinman.recording import with_recording
from tinman.stats import with_stats
from tinman.logging import with_log_capture
from tinman.pubsub import In, Out
from tinman.runtime import get_node_specs


# Test nodes
async def producer(output: Annotated[Out[str], "messages"]):
    """Producer node for testing."""
    for i in range(3):
        await output.publish(f"msg {i}")


async def consumer(input: Annotated[In[str], "messages"]):
    """Consumer node for testing."""
    async for msg in input:
        pass


class TestGraph:
    """Tests for Graph class."""
    
    def test_of_creates_graph_with_user_names(self):
        """Test Graph.of() creates graph with correct user node names."""
        graph = Graph.of(producer, consumer)
        
        assert len(graph.nodes) == 2
        assert graph.user_node_names == {"producer", "consumer"}
        assert graph.timer is None
    
    def test_to_dot_generates_valid_dot(self):
        """Test that to_dot() generates valid DOT output."""
        graph = Graph.of(producer, consumer)
        dot = graph.to_dot()
        
        assert "digraph tinman_graph" in dot
        assert "producer" in dot
        assert "consumer" in dot
        assert "messages" in dot
        assert "subgraph cluster_user" in dot


class TestWithStats:
    """Tests for with_stats transform."""
    
    def test_adds_stats_node(self):
        """Test with_stats adds a stats node."""
        graph = Graph.of(producer, consumer)
        assert len(graph.nodes) == 2
        
        graph = with_stats()(graph)
        
        # Should have 3 nodes now (producer, consumer, stats)
        assert len(graph.nodes) == 3
        # User nodes unchanged
        assert graph.user_node_names == {"producer", "consumer"}
    
    def test_stats_node_is_system_node(self):
        """Test that stats node appears in system cluster in DOT."""
        graph = Graph.of(producer, consumer)
        graph = with_stats()(graph)
        
        dot = graph.to_dot()
        
        # Stats node should be in system cluster
        assert "subgraph cluster_system" in dot
        assert "stats_node" in dot


class TestWithRecording:
    """Tests for with_recording transform."""
    
    def test_adds_logging_node(self):
        """Test with_recording adds a recording node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            graph = Graph.of(producer, consumer)
            assert len(graph.nodes) == 2
            
            graph = with_recording(log_dir)(graph)
            
            # Should have 3 nodes now
            assert len(graph.nodes) == 3
            # User nodes unchanged
            assert graph.user_node_names == {"producer", "consumer"}


class TestWithLogCapture:
    """Tests for with_log_capture transform."""
    
    def test_adds_log_capture_nodes(self):
        """Test with_log_capture adds capture and printer nodes."""
        graph = Graph.of(producer, consumer)
        
        graph = with_log_capture(print_logs=True)(graph)
        
        # Should have added 2 nodes (capture + printer)
        assert len(graph.nodes) == 4
        # Log handler should be stored
        assert graph._log_handler is not None


class TestGraphChaining:
    """Tests for chaining multiple transforms."""
    
    def test_chain_multiple_transforms(self):
        """Test that transforms can be chained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            graph = Graph.of(producer, consumer)
            graph = with_stats()(graph)
            graph = with_recording(log_dir)(graph)
            
            # Should have producer, consumer, stats, logging = 4 nodes
            assert len(graph.nodes) == 4
            # User nodes unchanged
            assert graph.user_node_names == {"producer", "consumer"}
    
    def test_dot_shows_all_system_nodes(self):
        """Test DOT output shows all system nodes in system cluster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            graph = Graph.of(producer, consumer)
            graph = with_stats()(graph)
            graph = with_recording(log_dir)(graph)
            
            dot = graph.to_dot()
            
            # Both system nodes should be in system cluster
            assert "subgraph cluster_system" in dot
            assert "stats_node" in dot


class TestGraphRun:
    """Tests for Graph.run() execution."""
    
    @pytest.mark.asyncio
    async def test_run_executes_nodes(self):
        """Test that run() actually executes nodes."""
        received = []
        
        async def test_producer(output: Annotated[Out[str], "data"]):
            await output.publish("hello")
        
        async def test_consumer(input: Annotated[In[str], "data"]):
            async for msg in input:
                received.append(msg)
        
        graph = Graph.of(test_producer, test_consumer)
        await graph.run()
        
        assert received == ["hello"]
