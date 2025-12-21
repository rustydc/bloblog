"""Virtual time system for deterministic fast-forward playback.

This module provides timers that coordinate time across all nodes.
Three execution modes are supported:

1. **Real-time** (speed=1.0): Wall clock time, actual delays
2. **Scaled playback** (speed≠1.0): Faster or slower than real-time
3. **Fast-forward** (speed=inf): Deterministic, as fast as possible

Key components:
- Timer: Protocol that nodes use to access time
- ScaledTimer: Wall clock with optional speed scaling
- FastForwardTimer: Virtual clock driven by event queue
- VirtualClock: Event-driven coordinator for fast-forward mode
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import time_ns
from typing import Protocol, runtime_checkable


# =============================================================================
# Timer Protocol
# =============================================================================


@runtime_checkable
class Timer(Protocol):
    """Protocol for time access in nodes.

    Nodes request a Timer parameter and the runtime injects the appropriate
    implementation based on execution mode.

    Example:
        >>> async def my_node(
        ...     input: Annotated[In[str], "data"],
        ...     timer: Timer,
        ... ) -> None:
        ...     async for item in input:
        ...         print(f"At {timer.time_ns()}: {item}")
        ...         await timer.sleep(0.1)
    """

    def time_ns(self) -> int:
        """Get current time in nanoseconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""
        ...

    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until the specified timestamp (no-op if already past)."""
        ...

    def periodic(self, period: float) -> AsyncIterator[int]:
        """Return an async iterator that yields timestamps at regular intervals.

        The first yield happens after one period has elapsed.
        """
        ...


# =============================================================================
# Base Implementation
# =============================================================================


class TimerBase(ABC):
    """Base class for Timer implementations with shared periodic logic."""

    @abstractmethod
    def time_ns(self) -> int:
        """Get current time in nanoseconds."""
        ...

    @abstractmethod
    async def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""
        ...

    @abstractmethod
    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until the specified timestamp."""
        ...

    async def periodic(self, period: float) -> AsyncIterator[int]:
        """Yield timestamps at regular intervals.

        Uses sleep_until for precise timing that doesn't drift.
        Each iteration sleeps until the next tick, then yields the current time.

        Args:
            period: Interval between yields in seconds.

        Yields:
            Current timestamp in nanoseconds after each period.

        Example:
            >>> async for t in timer.periodic(1.0):
            ...     print(f"Tick at {t}")
            ...     if should_stop:
            ...         break
        """
        period_ns = int(period * 1_000_000_000)
        next_tick = self.time_ns() + period_ns

        while True:
            await self.sleep_until(next_tick)
            yield self.time_ns()
            next_tick += period_ns


# =============================================================================
# Virtual Clock (event-driven coordinator for fast-forward mode)
# =============================================================================


@dataclass(order=True)
class _ScheduledEvent:
    """A pending event in the virtual clock's priority queue."""

    timestamp: int
    seq: int = field(compare=True)  # Tiebreaker for stable ordering
    future: asyncio.Future = field(compare=False)


class VirtualClock:
    """Event-driven coordinator for fast-forward playback.

    The clock maintains a priority queue of scheduled events. Each event
    has a timestamp and an asyncio Future. When schedule() is called,
    it adds the event to the queue, then pops and wakes the earliest
    event (which might be itself).

    This creates a self-driving system: each schedule() call advances
    the clock one step. The chain continues as woken tasks do their
    work and call schedule() again.

    Events are scheduled by:
    - FastForwardTimer.sleep_until() - for timer.sleep() calls
    - PlaybackIn.__anext__() - for message delivery from recorded logs

    This ensures deterministic execution: events are processed in
    timestamp order, and the clock only moves forward.

    Example:
        >>> clock = VirtualClock(start_time=1_000_000_000)
        >>> timer = FastForwardTimer(clock)
        >>>
        >>> # Events drive the clock:
        >>> await timer.sleep(1.0)  # Schedules event at T+1s
    """

    def __init__(self, start_time: int = 0):
        self._time: int = start_time
        self._events: list[_ScheduledEvent] = []
        self._seq: int = 0

    def now(self) -> int:
        """Get current virtual time in nanoseconds."""
        return self._time

    async def schedule(self, timestamp: int) -> None:
        """Schedule an event at the given timestamp and wait for it.

        The event is added to the priority queue, then we yield to let other
        tasks schedule their events. After yielding, the earliest event is
        popped and woken.

        This creates a self-driving clock: each schedule() advances one step,
        and the woken task will eventually call schedule() again, continuing
        the chain.

        Args:
            timestamp: Virtual time (ns) when this event should fire.
        """
        # Add ourselves to the queue
        future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        event = _ScheduledEvent(timestamp=timestamp, seq=self._seq, future=future)
        self._seq += 1
        heappush(self._events, event)

        # Yield to let other tasks schedule their events
        await asyncio.sleep(0)

        # Advance one step: pop and wake the earliest event
        if self._events:
            earliest = heappop(self._events)
            self._time = max(self._time, earliest.timestamp)
            earliest.future.set_result(None)

        # Wait for our turn (might already be done if we were earliest)
        await future

    def pending_count(self) -> int:
        """Number of pending events in the queue."""
        return len(self._events)

    def next_timestamp(self) -> int | None:
        """Timestamp of next pending event, or None if queue is empty."""
        return self._events[0].timestamp if self._events else None

    async def flush(self) -> None:
        """Process all pending events immediately.

        This is useful at the end of playback to ensure any timers scheduled
        by consumers are fired before the graph shuts down. All events are
        processed in timestamp order.

        After flush completes, pending_count() will be 0.
        """
        while self._events:
            earliest = heappop(self._events)
            self._time = max(self._time, earliest.timestamp)
            earliest.future.set_result(None)
            await asyncio.sleep(0)  # Let woken task run


# =============================================================================
# Timer Implementations
# =============================================================================


class ScaledTimer(TimerBase):
    """Timer with optional speed scaling.

    Maintains virtual time that advances at `speed` times real time.
    With speed=1.0, this behaves identically to wall clock time.

    Args:
        speed: Playback speed multiplier.
            - 1.0: real-time (virtual time == wall time)
            - 2.0: 2x speed (sleeps are halved)
            - 0.5: half speed (sleeps are doubled)
        start_time: Initial virtual time in nanoseconds. If None, uses
            current wall clock time.

    Example:
        >>> timer = ScaledTimer(speed=2.0)  # 2x playback
        >>> await timer.sleep(1.0)  # Actually sleeps 0.5 seconds
    """

    def __init__(self, speed: float = 1.0, start_time: int | None = None):
        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}")
        self._speed = speed
        self._start_real = time_ns()
        self._start_virtual = start_time if start_time is not None else self._start_real

    def time_ns(self) -> int:
        """Get current virtual time based on scaled elapsed real time."""
        elapsed_real = time_ns() - self._start_real
        elapsed_virtual = int(elapsed_real * self._speed)
        return self._start_virtual + elapsed_virtual

    async def sleep(self, seconds: float) -> None:
        """Sleep for the specified virtual duration."""
        if seconds <= 0:
            return
        real_seconds = seconds / self._speed
        await asyncio.sleep(real_seconds)

    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until virtual time reaches target."""
        current = self.time_ns()
        if target_ns > current:
            delay_virtual_ns = target_ns - current
            delay_real_s = delay_virtual_ns / self._speed / 1_000_000_000
            await asyncio.sleep(delay_real_s)


class FastForwardTimer(TimerBase):
    """Timer for fast-forward playback using VirtualClock.

    All operations delegate to the virtual clock's event queue. Time only
    advances when events are processed by advance_next().

    This ensures deterministic execution order: events are processed
    in timestamp order across all PlaybackIn channels and timer sleeps.

    Args:
        clock: The VirtualClock instance that coordinates time.

    Example:
        >>> clock = VirtualClock()
        >>> timer = FastForwardTimer(clock)
        >>>
        >>> # Timer blocks until clock advances
        >>> await timer.sleep(1.0)
    """

    def __init__(self, clock: VirtualClock):
        self._clock = clock

    def time_ns(self) -> int:
        """Get current virtual time from the clock."""
        return self._clock.now()

    async def sleep(self, seconds: float) -> None:
        """Schedule an event and wait for clock to reach it."""
        if seconds <= 0:
            return
        delay_ns = int(seconds * 1_000_000_000)
        target = self._clock.now() + delay_ns
        await self._clock.schedule(target)

    async def sleep_until(self, target_ns: int) -> None:
        """Schedule an event at target time and wait."""
        if target_ns <= self._clock.now():
            return
        await self._clock.schedule(target_ns)


# =============================================================================
# Factory
# =============================================================================


def create_timer(
    speed: float = 1.0,
    clock: VirtualClock | None = None,
    start_time: int | None = None,
) -> TimerBase:
    """Create appropriate timer for the execution mode.

    Args:
        speed: Playback speed multiplier.
            - 1.0: real-time
            - 2.0: 2x speed
            - 0.5: half speed
            - float('inf'): fast-forward (requires clock)
        clock: VirtualClock for fast-forward mode. Required if speed=inf.
        start_time: Initial virtual time for ScaledTimer. If None, uses
            current wall clock time.

    Returns:
        Timer instance appropriate for the execution mode.

    Raises:
        ValueError: If speed=inf but no clock provided, or speed <= 0.

    Example:
        >>> # Real-time execution
        >>> timer = create_timer()
        >>>
        >>> # 2x playback
        >>> timer = create_timer(speed=2.0)
        >>>
        >>> # Fast-forward with virtual clock
        >>> clock = VirtualClock()
        >>> timer = create_timer(speed=float('inf'), clock=clock)
    """
    if speed == float("inf"):
        if clock is None:
            raise ValueError("Fast-forward mode (speed=inf) requires a VirtualClock")
        return FastForwardTimer(clock)
    else:
        return ScaledTimer(speed, start_time)
