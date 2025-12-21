"""Tests for the CLI module."""

import sys
from pathlib import Path

import pytest

# Ensure tests directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinman.cli import load_nodes


# Define test nodes in this module - these are what we'll load
async def sample_node_a():
    """A simple test node."""
    pass


async def sample_node_b():
    """Another simple test node."""
    pass


sample_node_list = [sample_node_a, sample_node_b]


# Factory functions for testing
def create_single_node():
    """Factory that returns a single node."""
    return sample_node_a


def create_node_list():
    """Factory that returns a list of nodes."""
    return [sample_node_a, sample_node_b]


def factory_with_optional_args(config: str = "default"):
    """Factory with optional args should still be called."""
    return sample_node_a


def factory_with_required_args(config: str):
    """Factory with required args - should be treated as a node itself."""
    return sample_node_a


class TestLoadNodes:
    def test_load_single_node(self):
        nodes = load_nodes(("tests.cli_test:sample_node_a",))
        assert len(nodes) == 1
        assert nodes[0].__name__ == "sample_node_a"

    def test_load_multiple_nodes_comma_separated(self):
        nodes = load_nodes(("tests.cli_test:sample_node_a,sample_node_b",))
        assert len(nodes) == 2
        assert nodes[0].__name__ == "sample_node_a"
        assert nodes[1].__name__ == "sample_node_b"

    def test_load_multiple_specs(self):
        nodes = load_nodes(("tests.cli_test:sample_node_a", "tests.cli_test:sample_node_b"))
        assert len(nodes) == 2

    def test_load_node_list(self):
        """Loading a list variable should expand it."""
        nodes = load_nodes(("tests.cli_test:sample_node_list",))
        assert len(nodes) == 2
        assert nodes[0].__name__ == "sample_node_a"
        assert nodes[1].__name__ == "sample_node_b"

    def test_missing_module_raises(self):
        with pytest.raises(ModuleNotFoundError, match="nonexistent_module"):
            load_nodes(("nonexistent_module:foo",))

    def test_missing_attribute_raises(self):
        with pytest.raises(AttributeError, match="nonexistent"):
            load_nodes(("tests.cli_test:nonexistent",))

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Missing node names"):
            load_nodes(("tests.cli_test",))

    def test_empty_nodes_raises(self):
        with pytest.raises(ValueError, match="No nodes specified"):
            load_nodes(())

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="not callable"):
            load_nodes(("tests.cli_test:__name__",))

    def test_factory_returning_single_node(self):
        """A no-arg factory returning a node should be called."""
        nodes = load_nodes(("tests.cli_test:create_single_node",))
        assert len(nodes) == 1
        assert nodes[0] is sample_node_a

    def test_factory_returning_node_list(self):
        """A no-arg factory returning a list should be called and expanded."""
        nodes = load_nodes(("tests.cli_test:create_node_list",))
        assert len(nodes) == 2
        assert nodes[0] is sample_node_a
        assert nodes[1] is sample_node_b

    def test_factory_with_optional_args(self):
        """A factory with only optional args should be called."""
        nodes = load_nodes(("tests.cli_test:factory_with_optional_args",))
        assert len(nodes) == 1
        assert nodes[0] is sample_node_a

    def test_factory_with_required_args_treated_as_node(self):
        """A callable with required args is treated as a node, not a factory."""
        nodes = load_nodes(("tests.cli_test:factory_with_required_args",))
        assert len(nodes) == 1
        # It's the factory itself, not called
        assert nodes[0] is factory_with_required_args


class TestParseDuration:
    """Tests for the _parse_duration function."""
    
    def test_plain_number(self):
        from tinman.cli import _parse_duration
        assert _parse_duration("10") == 10.0
        assert _parse_duration("1.5") == 1.5
        assert _parse_duration("0.25") == 0.25
    
    def test_seconds(self):
        from tinman.cli import _parse_duration
        assert _parse_duration("10s") == 10.0
        assert _parse_duration("1.5sec") == 1.5
        assert _parse_duration("2secs") == 2.0
        assert _parse_duration("3second") == 3.0
        assert _parse_duration("4seconds") == 4.0
    
    def test_minutes(self):
        from tinman.cli import _parse_duration
        assert _parse_duration("1m") == 60.0
        assert _parse_duration("1.5min") == 90.0
        assert _parse_duration("2mins") == 120.0
        assert _parse_duration("3minute") == 180.0
        assert _parse_duration("4minutes") == 240.0
    
    def test_hours(self):
        from tinman.cli import _parse_duration
        assert _parse_duration("1h") == 3600.0
        assert _parse_duration("1.5hr") == 5400.0
        assert _parse_duration("2hrs") == 7200.0
        assert _parse_duration("3hour") == 10800.0
        assert _parse_duration("4hours") == 14400.0
    
    def test_whitespace(self):
        from tinman.cli import _parse_duration
        assert _parse_duration("  10s  ") == 10.0
        assert _parse_duration("5 m") == 300.0
    
    def test_case_insensitive(self):
        from tinman.cli import _parse_duration
        assert _parse_duration("10S") == 10.0
        assert _parse_duration("5M") == 300.0
        assert _parse_duration("1H") == 3600.0
    
    def test_invalid_format_raises(self):
        from tinman.cli import _parse_duration
        with pytest.raises(ValueError, match="Invalid duration format"):
            _parse_duration("abc")
        with pytest.raises(ValueError, match="Invalid duration format"):
            _parse_duration("10x")
        with pytest.raises(ValueError, match="Invalid duration format"):
            _parse_duration("ten seconds")