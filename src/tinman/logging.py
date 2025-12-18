"""Python logging integration for tinman.

This module provides a logging handler that captures Python log records
and outputs them as a tinman channel.

Usage:
    # Option 1: Use the handler directly
    handler = LogHandler()
    logging.getLogger().addHandler(handler)
    
    # Include handler.node in your graph
    await run([producer, consumer, handler.node])
    
    # Option 2: Use create_log_capture_node() factory
    handler, node = create_log_capture_node()
    logging.getLogger("myapp").addHandler(handler)
    await run([producer, consumer, node])
    
    # Option 3: CLI integration (automatic root logger)
    tinman run myapp:nodes --log-dir logs/ --capture-logs
    
    # Option 4: Virtual time integration (for playback)
    install_timer_log_factory(timer)
    # Now all log records use timer.time_ns() for timestamps

The log records are published as LogEntry dataclasses which can be
serialized efficiently without pickle.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Callable

from .oblog import Codec
from .pubsub import Out
from .runtime import NodeSpec
from .timer import Timer


@dataclass
class LogEntry:
    """Structured log entry for serialization.
    
    This captures the essential fields from logging.LogRecord in a
    format that's safe and efficient to serialize.
    
    Attributes:
        timestamp_ns: Unix timestamp in nanoseconds (from record.created)
        level: Log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50)
        name: Logger name (e.g., "myapp.module")
        message: Formatted log message
        pathname: Source file path (may be None)
        lineno: Line number in source file
        func_name: Function name (may be None)
        exc_text: Formatted exception text (may be None)
    """
    timestamp_ns: int
    level: int
    name: str
    message: str
    pathname: str | None = None
    lineno: int = 0
    func_name: str | None = None
    exc_text: str | None = None
    
    @classmethod
    def from_record(cls, record: logging.LogRecord) -> LogEntry:
        """Create LogEntry from a logging.LogRecord.
        
        Args:
            record: Python logging record to convert.
            
        Returns:
            LogEntry with fields extracted from the record.
        """
        # Format the message (applies % formatting if args present)
        message = record.getMessage()
        
        # Format exception if present
        exc_text = None
        if record.exc_info:
            import traceback
            exc_text = "".join(traceback.format_exception(*record.exc_info))
        elif record.exc_text:
            exc_text = record.exc_text
            
        return cls(
            timestamp_ns=int(record.created * 1_000_000_000),
            level=record.levelno,
            name=record.name,
            message=message,
            pathname=record.pathname,
            lineno=record.lineno,
            func_name=record.funcName,
            exc_text=exc_text,
        )
    
    def format(self, fmt: str = "%(levelname)s:%(name)s:%(message)s") -> str:
        """Format the log entry as a string.
        
        Args:
            fmt: Format string using logging-style placeholders.
            
        Returns:
            Formatted log string.
        """
        level_name = logging.getLevelName(self.level)
        result = fmt % {
            "levelname": level_name,
            "name": self.name,
            "message": self.message,
            "pathname": self.pathname or "",
            "lineno": self.lineno,
            "funcName": self.func_name or "",
        }
        if self.exc_text:
            result = f"{result}\n{self.exc_text}"
        return result


class LogEntryCodec(Codec[LogEntry]):
    """Efficient codec for LogEntry serialization.
    
    Binary format:
        - timestamp_ns: int64 (8 bytes)
        - level: int32 (4 bytes)
        - lineno: int32 (4 bytes)
        - name_len: uint32 (4 bytes)
        - message_len: uint32 (4 bytes)
        - pathname_len: uint32 (4 bytes, 0 = None)
        - func_name_len: uint32 (4 bytes, 0 = None)
        - exc_text_len: uint32 (4 bytes, 0 = None)
        - name: bytes
        - message: bytes
        - pathname: bytes (if present)
        - func_name: bytes (if present)
        - exc_text: bytes (if present)
    """
    
    _HEADER = struct.Struct("<qiiIIIII")  # 40 bytes header
    
    def encode(self, item: LogEntry) -> bytes:
        name_bytes = item.name.encode("utf-8")
        message_bytes = item.message.encode("utf-8")
        pathname_bytes = item.pathname.encode("utf-8") if item.pathname else b""
        func_name_bytes = item.func_name.encode("utf-8") if item.func_name else b""
        exc_text_bytes = item.exc_text.encode("utf-8") if item.exc_text else b""
        
        header = self._HEADER.pack(
            item.timestamp_ns,
            item.level,
            item.lineno,
            len(name_bytes),
            len(message_bytes),
            len(pathname_bytes),
            len(func_name_bytes),
            len(exc_text_bytes),
        )
        
        return header + name_bytes + message_bytes + pathname_bytes + func_name_bytes + exc_text_bytes
    
    def decode(self, data: bytes | memoryview) -> LogEntry:
        if isinstance(data, memoryview):
            data = bytes(data)
            
        (
            timestamp_ns,
            level,
            lineno,
            name_len,
            message_len,
            pathname_len,
            func_name_len,
            exc_text_len,
        ) = self._HEADER.unpack_from(data, 0)
        
        offset = self._HEADER.size
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        message = data[offset:offset + message_len].decode("utf-8")
        offset += message_len
        pathname = data[offset:offset + pathname_len].decode("utf-8") if pathname_len else None
        offset += pathname_len
        func_name = data[offset:offset + func_name_len].decode("utf-8") if func_name_len else None
        offset += func_name_len
        exc_text = data[offset:offset + exc_text_len].decode("utf-8") if exc_text_len else None
        
        return LogEntry(
            timestamp_ns=timestamp_ns,
            level=level,
            name=name,
            message=message,
            pathname=pathname,
            lineno=lineno,
            func_name=func_name,
            exc_text=exc_text,
        )


class LogHandler(logging.Handler):
    """Logging handler that queues records for a tinman node.
    
    This handler captures log records and puts them in an asyncio queue.
    The associated node reads from the queue and publishes LogEntry objects.
    
    When `use_virtual_time=True`, the node requests Timer injection and
    installs the timer log factory, so all Python log records use the
    injected timer's time (useful for playback with virtual time).
    
    Example:
        >>> handler = LogHandler(channel="logs", level=logging.INFO)
        >>> logging.getLogger().addHandler(handler)
        >>> 
        >>> # Include in your node graph
        >>> await run([my_producer, my_consumer, handler.node])
        >>> 
        >>> # With virtual time during playback
        >>> handler = LogHandler(channel="logs", use_virtual_time=True)
        >>> logging.getLogger().addHandler(handler)
        >>> await playback([my_consumer, handler.node], Path("logs"))
    
    Attributes:
        channel: Name of the output channel (default: "logs")
        node: The NodeSpec to include in your graph
        use_virtual_time: Whether to use injected Timer for log timestamps
    """
    
    def __init__(
        self,
        channel: str = "logs",
        level: int = logging.NOTSET,
        maxsize: int = 1000,
        use_virtual_time: bool = False,
    ):
        """Initialize the log handler.
        
        Args:
            channel: Output channel name for log entries.
            level: Minimum log level to capture (default: NOTSET = all levels).
            maxsize: Maximum queue size before blocking (default: 1000).
            use_virtual_time: If True, request Timer injection and use it for
                log timestamps. This is useful during playback to have log
                messages use virtual/scaled time. Default: False.
        """
        super().__init__(level)
        self.channel = channel
        self.use_virtual_time = use_virtual_time
        self._queue: asyncio.Queue[LogEntry | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        
        # Create the node as a NodeSpec to avoid annotation evaluation issues
        self.node = self._create_node_spec()
    
    def _create_node_spec(self) -> NodeSpec:
        """Create the NodeSpec for this handler."""
        queue = self._queue
        channel = self.channel
        use_virtual_time = self.use_virtual_time
        
        async def log_capture_node(output: Out[LogEntry]) -> None:
            """Node that publishes captured log records."""
            while True:
                entry = await queue.get()
                if entry is None:
                    break
                await output.publish(entry)
        
        async def log_capture_node_with_timer(output: Out[LogEntry], timer: Timer) -> None:
            """Node that publishes captured log records with virtual time."""
            # Install timer log factory for virtual timestamps
            install_timer_log_factory(timer)
            try:
                while True:
                    entry = await queue.get()
                    if entry is None:
                        break
                    await output.publish(entry)
            finally:
                uninstall_timer_log_factory()
        
        # Choose the appropriate node function
        if use_virtual_time:
            node_fn = log_capture_node_with_timer
            timer_param = "timer"
        else:
            node_fn = log_capture_node
            timer_param = None
        
        # Set a nice name for debugging
        node_fn.__name__ = f"log_capture[{channel}]"
        node_fn.__qualname__ = node_fn.__name__
        
        return NodeSpec(
            node_fn=node_fn,
            inputs={},
            outputs={"output": (channel, LogEntryCodec())},
            all_channels_param=None,
            timer_param=timer_param,
            daemon=True,  # Don't block shutdown - cancel when main nodes complete
        )
    
    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record by queueing it for the node.
        
        This is called by the logging framework. It converts the record
        to a LogEntry and puts it in the queue.
        
        Args:
            record: Log record from the logging framework.
        """
        if self._closed:
            return
            
        try:
            entry = LogEntry.from_record(record)
            # Use put_nowait to avoid blocking the logging call
            # If queue is full, we drop the log entry (better than blocking)
            try:
                self._queue.put_nowait(entry)
            except asyncio.QueueFull:
                # Queue full - drop this entry to avoid blocking
                pass
        except Exception:
            # Don't let logging errors crash the application
            self.handleError(record)
    
    def close(self) -> None:
        """Close the handler and signal the node to stop."""
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        super().close()


def create_log_capture_node(
    channel: str = "logs",
    level: int = logging.NOTSET,
    logger: logging.Logger | str | None = None,
    maxsize: int = 1000,
) -> tuple[LogHandler, NodeSpec]:
    """Create a log handler and capture node pair.
    
    This is a convenience factory that creates both the handler and node,
    optionally attaching the handler to a logger.
    
    Args:
        channel: Output channel name for log entries.
        level: Minimum log level to capture (default: NOTSET = all levels).
        logger: Logger to attach handler to. Can be:
            - None: Don't attach (you'll attach manually)
            - str: Get logger by name and attach
            - Logger: Attach to this logger
        maxsize: Maximum queue size before dropping entries.
        
    Returns:
        Tuple of (handler, node) where:
            - handler: LogHandler to attach to loggers
            - node: NodeSpec to include in your node graph
    
    Example:
        >>> # Capture all logs from "myapp" logger
        >>> handler, node = create_log_capture_node(logger="myapp")
        >>> await run([producer, consumer, node], log_dir=Path("logs"))
        
        >>> # Manual handler attachment
        >>> handler, node = create_log_capture_node()
        >>> logging.getLogger("myapp.module1").addHandler(handler)
        >>> logging.getLogger("myapp.module2").addHandler(handler)
        >>> await run([producer, consumer, node])
    """
    handler = LogHandler(channel=channel, level=level, maxsize=maxsize)
    
    if logger is not None:
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        logger.addHandler(handler)
    
    return handler, handler.node


# =============================================================================
# Timer-aware LogRecord factory
# =============================================================================

# Store the original factory so we can restore it
_original_log_record_factory: Callable[..., logging.LogRecord] | None = None

# Global timer reference for the factory
_current_timer: Timer | None = None


class TimerLogRecord(logging.LogRecord):
    """LogRecord that uses a Timer for timestamps instead of time.time().
    
    This allows log records to use virtual time during playback, ensuring
    that logged timestamps match the simulation time rather than wall clock.
    """
    
    def __init__(
        self,
        name: str,
        level: int,
        pathname: str,
        lineno: int,
        msg: object,
        args: tuple,
        exc_info: tuple | None,
        func: str | None = None,
        sinfo: str | None = None,
    ):
        # Get time from timer before calling super().__init__
        # which would set self.created from time.time()
        if _current_timer is not None:
            timer_time_ns = _current_timer.time_ns()
            timer_time = timer_time_ns / 1_000_000_000
        else:
            timer_time = None
        
        super().__init__(
            name, level, pathname, lineno, msg, args, exc_info, func, sinfo
        )
        
        # Override the timestamp if we have a timer
        if timer_time is not None:
            self.created = timer_time
            self.msecs = (timer_time - int(timer_time)) * 1000
            # logging._startTime is the time when the logging module was loaded
            start_time: float = getattr(logging, "_startTime", self.created)
            self.relativeCreated = (timer_time - start_time) * 1000


def _timer_log_record_factory(
    name: str,
    level: int,
    pathname: str,
    lineno: int,
    msg: object,
    args: tuple,
    exc_info: tuple | None,
    func: str | None = None,
    sinfo: str | None = None,
) -> logging.LogRecord:
    """Factory function that creates TimerLogRecord instances."""
    return TimerLogRecord(name, level, pathname, lineno, msg, args, exc_info, func, sinfo)


def install_timer_log_factory(timer: Timer) -> None:
    """Install a log record factory that uses the given Timer for timestamps.
    
    After calling this, all new LogRecord instances will use `timer.time_ns()`
    for their `created` timestamp instead of `time.time()`.
    
    This is useful during playback to ensure log timestamps match virtual time.
    
    Args:
        timer: The Timer instance to use for timestamps.
        
    Example:
        >>> from tinman import VirtualClock, FastForwardTimer
        >>> from tinman.logging import install_timer_log_factory
        >>> 
        >>> clock = VirtualClock(start_time=1000000000000)  # 1 second in ns
        >>> timer = FastForwardTimer(clock)
        >>> install_timer_log_factory(timer)
        >>> 
        >>> # Now logging uses virtual time
        >>> logging.info("This message has virtual timestamp")
        
    Note:
        Call `uninstall_timer_log_factory()` to restore normal behavior.
        Or use `timer_log_context(timer)` as a context manager.
    """
    global _original_log_record_factory, _current_timer
    
    if _original_log_record_factory is None:
        _original_log_record_factory = logging.getLogRecordFactory()
    
    _current_timer = timer
    logging.setLogRecordFactory(_timer_log_record_factory)


def uninstall_timer_log_factory() -> None:
    """Restore the original log record factory.
    
    Call this after playback is complete to restore normal timestamp behavior.
    """
    global _original_log_record_factory, _current_timer
    
    if _original_log_record_factory is not None:
        logging.setLogRecordFactory(_original_log_record_factory)
        _original_log_record_factory = None
    
    _current_timer = None


@contextmanager
def timer_log_context(timer: Timer):
    """Context manager that temporarily installs a timer-aware log factory.
    
    Use this to ensure log records use virtual time within a specific scope,
    automatically restoring the original factory on exit.
    
    Args:
        timer: The Timer instance to use for timestamps.
        
    Yields:
        None
        
    Example:
        >>> with timer_log_context(timer):
        ...     logging.info("Uses virtual time")
        >>> logging.info("Uses wall clock time again")
    """
    install_timer_log_factory(timer)
    try:
        yield
    finally:
        uninstall_timer_log_factory()


def get_current_timer() -> Timer | None:
    """Get the currently installed timer, if any.
    
    Returns:
        The Timer passed to install_timer_log_factory(), or None if
        no timer is installed.
    """
    return _current_timer


@contextmanager
def log_capture_context(
    enabled: bool = True,
    channel: str = "logs",
    level: int = logging.INFO,
    use_virtual_time: bool = False,
):
    """Context manager for capturing Python logs to a tinman channel.
    
    Sets up a LogHandler on the root logger and yields a list containing
    the handler's node. If disabled, yields an empty list.
    
    Args:
        enabled: If False, yields empty list and does nothing. Default: True.
        channel: Channel name for log output. Default: "logs".
        level: Minimum log level to capture. Default: logging.INFO.
        use_virtual_time: If True, the node will request Timer injection
            and use virtual time for log timestamps. Default: False.
            
    Yields:
        List containing the handler's node, or empty list if disabled.
        
    Example:
        >>> with log_capture_context(capture_logs) as log_nodes:
        ...     nodes = [producer, consumer, *log_nodes]
        ...     asyncio.run(run(nodes, log_dir="logs/"))
    """
    if not enabled:
        yield []
        return
    
    handler = LogHandler(
        channel=channel,
        level=level,
        use_virtual_time=use_virtual_time,
    )
    logging.getLogger().addHandler(handler)
    try:
        yield [handler.node]
    finally:
        handler.close()
        logging.getLogger().removeHandler(handler)


def create_log_printer(channel: str = "logs") -> NodeSpec:
    """Create a daemon node that prints log entries from a channel.
    
    Args:
        channel: Channel name to subscribe to. Default: "logs".
        
    Returns:
        A NodeSpec (daemon) that prints log entries with colored severity.
        
    Example:
        >>> # Auto-attach to print captured logs
        >>> nodes = [producer, consumer, log_handler.node, create_log_printer()]
        >>> await run(nodes)
    """
    from datetime import datetime
    from .pubsub import In
    from rich.console import Console
    from rich.text import Text
    
    # Color mapping for log levels
    LEVEL_STYLES = {
        logging.DEBUG: "dim",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red bold",
    }
    
    console = Console()
    
    async def log_printer_node(logs: In[LogEntry]) -> None:
        """Print log entries as they arrive."""
        async for entry in logs:
            level_name = logging.getLevelName(entry.level)
            style = LEVEL_STYLES.get(entry.level, "white")
            # Convert timestamp_ns to datetime
            timestamp = datetime.fromtimestamp(entry.timestamp_ns / 1_000_000_000)
            time_str = timestamp.strftime("%H:%M:%S.%f")
            
            text = Text()
            text.append(time_str, style="dim")
            text.append(" ")
            text.append(f"[{level_name:8}]", style=style)
            text.append(f" {entry.name}: {entry.message}")
            console.print(text)
            
            if entry.exc_text:
                console.print(Text(entry.exc_text, style="red"))
    
    return NodeSpec(
        node_fn=log_printer_node,
        inputs={"logs": (channel, 100)},  # Larger queue to avoid blocking
        outputs={},
        all_channels_param=None,
        timer_param=None,
        daemon=True,
    )


# Convenience for getting the codec
LOG_CODEC = LogEntryCodec()
