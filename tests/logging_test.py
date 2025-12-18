"""Tests for Python logging integration."""

import asyncio
import logging
from typing import Annotated

import pytest

from tinman import In, Out, run, VirtualClock, FastForwardTimer
from tinman.logging import (
    LogEntry,
    LogEntryCodec,
    LogHandler,
    create_log_capture_node,
    install_timer_log_factory,
    uninstall_timer_log_factory,
    timer_log_context,
    get_current_timer,
)


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_from_record_basic(self):
        """Test basic LogRecord conversion."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        
        entry = LogEntry.from_record(record)
        
        assert entry.name == "test.logger"
        assert entry.level == logging.INFO
        assert entry.message == "Hello world"
        assert entry.pathname == "/path/to/file.py"
        assert entry.lineno == 42
        assert entry.exc_text is None

    def test_from_record_with_exception(self):
        """Test LogRecord conversion with exception info."""
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=100,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )
        
        entry = LogEntry.from_record(record)
        
        assert entry.level == logging.ERROR
        assert entry.message == "An error occurred"
        assert entry.exc_text is not None
        assert "ValueError: test error" in entry.exc_text

    def test_format(self):
        """Test log entry formatting."""
        entry = LogEntry(
            timestamp_ns=1000000000,
            level=logging.WARNING,
            name="myapp.module",
            message="Something happened",
            pathname="/path/file.py",
            lineno=50,
        )
        
        formatted = entry.format()
        assert "WARNING" in formatted
        assert "myapp.module" in formatted
        assert "Something happened" in formatted

    def test_format_with_node_name(self):
        """Test formatting with node_name placeholder."""
        entry = LogEntry(
            timestamp_ns=1000000000,
            level=logging.INFO,
            name="myapp",
            message="Hello",
            node_name="my_producer",
        )
        
        formatted = entry.format("[%(node_name)s] %(levelname)s: %(message)s")
        assert formatted == "[my_producer] INFO: Hello"

    def test_from_record_captures_node_name(self):
        """Test that from_record captures the current node name."""
        from tinman.runtime import _current_node_name
        
        # Simulate being inside a node
        token = _current_node_name.set("test_node")
        try:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="/path/to/file.py",
                lineno=42,
                msg="Hello",
                args=(),
                exc_info=None,
            )
            
            entry = LogEntry.from_record(record)
            assert entry.node_name == "test_node"
        finally:
            _current_node_name.reset(token)

    def test_from_record_node_name_none_outside_node(self):
        """Test that node_name is None when not in a node context."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Hello",
            args=(),
            exc_info=None,
        )
        
        entry = LogEntry.from_record(record)
        assert entry.node_name is None


class TestLogEntryCodec:
    """Tests for LogEntryCodec serialization."""

    def test_encode_decode_basic(self):
        """Test basic encode/decode roundtrip."""
        codec = LogEntryCodec()
        entry = LogEntry(
            timestamp_ns=1234567890123456789,
            level=logging.DEBUG,
            name="test.codec",
            message="Test message",
            pathname="/test/path.py",
            lineno=123,
            func_name="test_function",
            exc_text=None,
            node_name="my_node",
        )
        
        encoded = codec.encode(entry)
        decoded = codec.decode(encoded)
        
        assert decoded.timestamp_ns == entry.timestamp_ns
        assert decoded.level == entry.level
        assert decoded.name == entry.name
        assert decoded.message == entry.message
        assert decoded.pathname == entry.pathname
        assert decoded.lineno == entry.lineno
        assert decoded.func_name == entry.func_name
        assert decoded.exc_text == entry.exc_text
        assert decoded.node_name == entry.node_name

    def test_encode_decode_with_none_fields(self):
        """Test encode/decode with optional fields as None."""
        codec = LogEntryCodec()
        entry = LogEntry(
            timestamp_ns=1000,
            level=logging.INFO,
            name="test",
            message="msg",
            pathname=None,
            lineno=0,
            func_name=None,
            exc_text=None,
            node_name=None,
        )
        
        decoded = codec.decode(codec.encode(entry))
        
        assert decoded.pathname is None
        assert decoded.func_name is None
        assert decoded.exc_text is None
        assert decoded.node_name is None

    def test_encode_decode_with_exception(self):
        """Test encode/decode with exception text."""
        codec = LogEntryCodec()
        entry = LogEntry(
            timestamp_ns=1000,
            level=logging.ERROR,
            name="test",
            message="error",
            exc_text="Traceback:\n  ValueError: oops",
        )
        
        decoded = codec.decode(codec.encode(entry))
        
        assert decoded.exc_text == "Traceback:\n  ValueError: oops"

    def test_encode_decode_unicode(self):
        """Test encode/decode with unicode characters."""
        codec = LogEntryCodec()
        entry = LogEntry(
            timestamp_ns=1000,
            level=logging.INFO,
            name="テスト",
            message="Hello 世界! 🎉",
        )
        
        decoded = codec.decode(codec.encode(entry))
        
        assert decoded.name == "テスト"
        assert decoded.message == "Hello 世界! 🎉"


class TestLogHandler:
    """Tests for LogHandler."""

    @pytest.mark.asyncio
    async def test_handler_captures_logs(self):
        """Test that handler captures log records."""
        handler = LogHandler(channel="test_logs")
        logger = logging.getLogger("test_capture")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            # Log some messages
            logger.info("Test message 1")
            logger.warning("Test message 2")
            
            # Check queue has entries
            assert handler._queue.qsize() == 2
            
            entry1 = handler._queue.get_nowait()
            assert entry1.message == "Test message 1"
            assert entry1.level == logging.INFO
            
            entry2 = handler._queue.get_nowait()
            assert entry2.message == "Test message 2"
            assert entry2.level == logging.WARNING
        finally:
            logger.removeHandler(handler)
            handler.close()

    @pytest.mark.asyncio
    async def test_handler_level_filter(self):
        """Test that handler respects level filter."""
        handler = LogHandler(channel="test_logs", level=logging.WARNING)
        logger = logging.getLogger("test_level")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            # Only WARNING and ERROR should be captured
            assert handler._queue.qsize() == 2
        finally:
            logger.removeHandler(handler)
            handler.close()

    @pytest.mark.asyncio
    async def test_handler_close_signals_node(self):
        """Test that close() signals the node to stop."""
        handler = LogHandler(channel="test_logs")
        
        handler.close()
        
        # Should have None in queue to signal stop
        item = handler._queue.get_nowait()
        assert item is None


class TestLogCaptureNode:
    """Integration tests for log capture node."""

    @pytest.mark.asyncio
    async def test_log_capture_in_graph(self):
        """Test log capture node in a full graph."""
        captured: list[LogEntry] = []
        
        handler = LogHandler(channel="app_logs")
        logger = logging.getLogger("test_graph")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        async def producer(output: Annotated[Out[str], "messages"]):
            for i in range(3):
                logger.info(f"Producing message {i}")
                await output.publish(f"msg-{i}")
            # Close handler when producer is done
            handler.close()
        
        async def log_consumer(logs: Annotated[In[LogEntry], "app_logs"]):
            async for entry in logs:
                captured.append(entry)
        
        try:
            await run([producer, handler.node, log_consumer])
            
            assert len(captured) == 3
            assert captured[0].message == "Producing message 0"
            assert captured[1].message == "Producing message 1"
            assert captured[2].message == "Producing message 2"
        finally:
            logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_create_log_capture_node_factory(self):
        """Test create_log_capture_node factory function."""
        handler, node = create_log_capture_node(
            channel="custom_logs",
            level=logging.WARNING,
            logger="test_factory",
        )
        
        logger = logging.getLogger("test_factory")
        logger.setLevel(logging.DEBUG)
        
        try:
            logger.info("Should not capture")
            logger.warning("Should capture")
            
            assert handler._queue.qsize() == 1
            entry = handler._queue.get_nowait()
            assert entry.message == "Should capture"
        finally:
            logger.removeHandler(handler)
            handler.close()


class TestTimerLogFactory:
    """Tests for timer-aware log record factory."""

    def test_install_uninstall(self):
        """Test installing and uninstalling the timer factory."""
        clock = VirtualClock(start_time=5_000_000_000)  # 5 seconds
        timer = FastForwardTimer(clock)
        
        # Initially no timer
        assert get_current_timer() is None
        
        install_timer_log_factory(timer)
        assert get_current_timer() is timer
        
        uninstall_timer_log_factory()
        assert get_current_timer() is None

    def test_logger_uses_timer_time(self):
        """Test that logger calls use the timer's time."""
        clock = VirtualClock(start_time=10_000_000_000)  # 10 seconds in ns
        timer = FastForwardTimer(clock)
        
        # Create a handler that captures records
        captured_records: list[logging.LogRecord] = []
        
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)
        
        logger = logging.getLogger("test_timer_basic")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)
        
        try:
            with timer_log_context(timer):
                logger.info("Message at virtual time 10s")
            
            assert len(captured_records) == 1
            # created should be 10.0 seconds
            assert captured_records[0].created == 10.0
        finally:
            logger.removeHandler(handler)

    def test_logger_uses_timer_time_with_fractional(self):
        """Test timer time with fractional seconds."""
        clock = VirtualClock(start_time=12_345_678_901)  # ~12.3 seconds
        timer = FastForwardTimer(clock)
        
        captured_records: list[logging.LogRecord] = []
        
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)
        
        logger = logging.getLogger("test_timer_fractional")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)
        
        try:
            with timer_log_context(timer):
                logger.info("test")
            
            # Should be approximately 12.345678901 seconds
            assert abs(captured_records[0].created - 12.345678901) < 0.0001
        finally:
            logger.removeHandler(handler)

    def test_context_manager(self):
        """Test timer_log_context context manager."""
        clock = VirtualClock(start_time=20_000_000_000)
        timer = FastForwardTimer(clock)
        
        assert get_current_timer() is None
        
        captured_records: list[logging.LogRecord] = []
        
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)
        
        logger = logging.getLogger("test_timer_context")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)
        
        try:
            with timer_log_context(timer):
                assert get_current_timer() is timer
                logger.info("test")
            
            assert captured_records[0].created == 20.0
            
            # After context, timer should be uninstalled
            assert get_current_timer() is None
        finally:
            logger.removeHandler(handler)

    def test_context_manager_restores_on_exception(self):
        """Test that context manager restores factory on exception."""
        clock = VirtualClock(start_time=1_000_000_000)
        timer = FastForwardTimer(clock)
        
        with pytest.raises(ValueError):
            with timer_log_context(timer):
                assert get_current_timer() is timer
                raise ValueError("test error")
        
        # Should be restored even after exception
        assert get_current_timer() is None

    def test_log_entry_from_logger_preserves_timer_time(self):
        """Test that LogEntry.from_record preserves timer timestamps from logger."""
        clock = VirtualClock(start_time=42_000_000_000)  # 42 seconds
        timer = FastForwardTimer(clock)
        
        captured_entries: list[LogEntry] = []
        
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_entries.append(LogEntry.from_record(record))
        
        logger = logging.getLogger("test_timer_entry")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)
        
        try:
            with timer_log_context(timer):
                logger.warning("warning message")
            
            # Should be 42 seconds in nanoseconds
            assert captured_entries[0].timestamp_ns == 42_000_000_000
        finally:
            logger.removeHandler(handler)

    def test_multiple_log_calls_use_current_timer_time(self):
        """Test that advancing the clock affects subsequent log timestamps."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        
        captured_records: list[logging.LogRecord] = []
        
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)
        
        logger = logging.getLogger("test_timer_advance")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)
        
        try:
            with timer_log_context(timer):
                logger.info("At time 0")  # t=0
                
                # Manually advance clock (simulating what playback does)
                import asyncio
                asyncio.run(clock.advance_to(5_000_000_000))  # 5 seconds
                
                logger.info("At time 5")  # t=5s
                
                asyncio.run(clock.advance_to(10_000_000_000))  # 10 seconds
                
                logger.info("At time 10")  # t=10s
            
            assert len(captured_records) == 3
            assert captured_records[0].created == 0.0
            assert captured_records[1].created == 5.0
            assert captured_records[2].created == 10.0
        finally:
            logger.removeHandler(handler)
