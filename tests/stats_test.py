"""Tests for the stats module."""

import io
from pathlib import Path

import pytest

from tinman.stats import ChannelStats, StatsCollector, create_stats_node, run_stats


class TestChannelStats:
    """Tests for ChannelStats class."""

    def test_record_message_first(self):
        """First message sets first_time_ns."""
        stats = ChannelStats(name="test")
        stats.record_message(1_000_000_000)
        
        assert stats.count == 1
        assert stats.first_time_ns == 1_000_000_000
        assert stats.last_time_ns == 1_000_000_000

    def test_record_message_multiple(self):
        """Multiple messages update count and last_time_ns."""
        stats = ChannelStats(name="test")
        stats.record_message(1_000_000_000)
        stats.record_message(2_000_000_000)
        stats.record_message(3_000_000_000)
        
        assert stats.count == 3
        assert stats.first_time_ns == 1_000_000_000
        assert stats.last_time_ns == 3_000_000_000

    def test_duration_seconds(self):
        """Duration is calculated correctly."""
        stats = ChannelStats(name="test")
        stats.record_message(1_000_000_000)  # 1 second
        stats.record_message(3_000_000_000)  # 3 seconds
        
        assert stats.duration_seconds == 2.0

    def test_duration_seconds_no_messages(self):
        """Duration is 0 with no messages."""
        stats = ChannelStats(name="test")
        assert stats.duration_seconds == 0.0

    def test_hz_calculation(self):
        """Hz is calculated as (count-1) / duration."""
        stats = ChannelStats(name="test")
        # 11 messages over 1 second = 10 Hz (10 intervals)
        for i in range(11):
            stats.record_message(i * 100_000_000)  # 0.1 second intervals
        
        assert stats.hz == pytest.approx(10.0)

    def test_hz_single_message(self):
        """Hz is 0 with only one message."""
        stats = ChannelStats(name="test")
        stats.record_message(1_000_000_000)
        assert stats.hz == 0.0


class TestStatsCollector:
    """Tests for StatsCollector class."""

    def test_record_new_channel(self):
        """Recording creates new channel stats."""
        collector = StatsCollector()
        collector.record("test", 1_000_000_000)
        
        assert "test" in collector.channels
        assert collector.channels["test"].count == 1

    def test_record_existing_channel(self):
        """Recording updates existing channel stats."""
        collector = StatsCollector()
        collector.record("test", 1_000_000_000)
        collector.record("test", 2_000_000_000)
        
        assert collector.channels["test"].count == 2

    def test_format_stats_empty(self):
        """Format returns message when no channels."""
        collector = StatsCollector()
        assert collector.format_stats() == "No channels recorded."

    def test_format_stats_with_data(self):
        """Format produces table with header and data."""
        collector = StatsCollector()
        collector.record("camera", 0)
        collector.record("camera", 100_000_000)
        collector.record("lidar", 0)
        
        output = collector.format_stats()
        
        assert "Channel" in output
        assert "Count" in output
        assert "Hz" in output
        assert "camera" in output
        assert "lidar" in output
        assert "2" in output  # camera count
        assert "1" in output  # lidar count

    def test_print_stats(self):
        """print_stats writes to specified file."""
        collector = StatsCollector()
        collector.record("test", 0)
        
        output = io.StringIO()
        collector.print_stats(file=output)
        
        assert "test" in output.getvalue()


class TestCreateStatsNode:
    """Tests for create_stats_node factory."""

    def test_creates_node_spec(self):
        """Returns a NodeSpec with correct configuration."""
        from tinman.runtime import NodeSpec
        
        node = create_stats_node()
        
        assert isinstance(node, NodeSpec)
        assert node.all_channels_param == "channels"
        assert node.timer_param == "timer"
        assert node.daemon is True

    def test_daemon_false(self):
        """daemon=False is respected."""
        node = create_stats_node(daemon=False)
        assert node.daemon is False


class TestRunStats:
    """Tests for run_stats function."""

    @pytest.mark.asyncio
    async def test_no_log_files(self, tmp_path: Path):
        """Reports no files when directory is empty."""
        output = io.StringIO()
        await run_stats(tmp_path, output=output)
        
        assert "No log files found" in output.getvalue()

    @pytest.mark.asyncio
    async def test_with_log_files(self, tmp_path: Path):
        """Reads and reports stats from log files."""
        from tinman import ObLogWriter
        from tinman.codecs import StringCodec
        
        # Create a test log
        async with ObLogWriter(tmp_path) as oblog:
            write = oblog.get_writer("test_channel", StringCodec())
            write("message1")
            write("message2")
            write("message3")
        
        # Run stats
        output = io.StringIO()
        await run_stats(tmp_path, output=output)
        
        result = output.getvalue()
        assert "test_channel" in result
        assert "3" in result  # count
